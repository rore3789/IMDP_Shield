
from juliacall import Main as jl
from generic_fnc import *
import cvxpy as cp
import yaml

jl.seval("using PosteriorBounds")
theta_vectors = jl.seval("PosteriorBounds.theta_vectors")

# exp_dir = os.path.dirname(os.path.abspath(__file__))
exp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')

des_loss = None
individual_nns = 0  # TODO, enable this as a user input and provide example

random_Seed = 20
nn_lr = 1e-4  # neural network learning rate for SGD
learning_rate = 0.001  # base kernel learning rate parameter with GP regression
epochs = 500  # how many update steps to optimize base kernel paramters

yaml_file_name = sys.argv[1]
yaml_file_name = exp_dir + yaml_file_name
with open(yaml_file_name, 'r') as file:
    yaml_data = yaml.safe_load(file)

global_dir_name = yaml_data["global_dir_name"]
process_dist = yaml_data["process_dist"]
pred_var = [(process_dist["sig"][dim] * 2.0) ** 2.0 for dim in range(len(process_dist["sig"]))]

unknown_modes_list = yaml_data["unknown_modes_list"]
Bs = yaml_data["Bs"]
GP_data_points = yaml_data["GP_data_points"]
kernel_data_points = yaml_data["kernel_data_points"]
nn_epochs = yaml_data["nn_epochs"]
width_1 = yaml_data["width_1"]
width_2 = yaml_data["width_2"]
layers = yaml_data["layers"]
reuse_data = yaml_data["reuse_data"]

X = yaml_data["X"]
grid_list = yaml_data["grid_list"]

keys = list(X)
d = len(keys)
alphas = [0.01 for _ in range(d)]

elw = [nn_epochs, nn_lr, 0]
network_dims = [d, width_1, width_2, d, layers]

using_gelu = False

global_exp_dir = exp_dir + "/Systems/" + global_dir_name

if os.path.isdir(exp_dir):
    if not os.path.isdir(global_exp_dir):
        if not os.path.isdir(exp_dir + "/Systems"):
            os.mkdir(exp_dir + "/Systems")
        os.mkdir(global_exp_dir)
else:
    os.mkdir(exp_dir)
    os.mkdir(exp_dir + "/Systems")
    os.mkdir(global_exp_dir)

modes = [i for i in range(len(unknown_modes_list))]
all_data = [[None, None] for _ in modes]
unknown_dyn_gp = [None for _ in modes]
region_info = [None for _ in modes]

# ==================================================================================================================== #
# ========================================== Generate/Load Training Data ============================================= #
# ==================================================================================================================== #

data_file_name = global_exp_dir + "/training_data.pkl"
if os.path.exists(data_file_name):
    print("Loading previous training data...")
    all_data = dict_load(data_file_name)
else:
    for mode in modes:
        g = unknown_modes_list[mode]
        if global_dir_name == "example_2D_sys":
            x_train, y_train = generate_training_data(g, X, GP_data_points, random_seed=random_Seed + mode,
                                                      process_dist=process_dist)
        else:
            # TODO, should be an example for sampling from a gym environment
            print("Please provide training data in the appropriate format, see the example data for information")
            exit()

        all_data[mode][0] = x_train
        all_data[mode][1] = y_train
    # don't carry around potentially large variables that are stored elsewhere
    del x_train, y_train
    dict_save(data_file_name, all_data)

# ==================================================================================================================== #
# ============================================= Deep Kernel Regression =============================================== #
# ==================================================================================================================== #

for mode in modes:
    unknown_dyn_gp[mode], region_info[mode] = deep_kernel_learning(all_data, mode, keys, network_dims, alphas, elw,
                                                                   lr=learning_rate, training_iter=epochs,
                                                                   random_seed=(mode + 2) * random_Seed,
                                                                   kernel_data_points=kernel_data_points,
                                                                   using_gelu=using_gelu, individual_nns=individual_nns,
                                                                   des_loss=des_loss)
    print(f'Finished deep kernel regression for mode {mode + 1}\n')

# ==================================================================================================================== #
# ========================== Get the posteriors of the NN over the discretization ==================================== #
# ==================================================================================================================== #
grid_len = {k: grid_list[k] for k in keys}

grid_size, large_grid = get_grid_info(X, grid_len)
extents = discretize_space_list(X, grid_size)
filename = global_exp_dir + f"/extents_0"
np.save(filename, np.array(extents))
extents.pop()

num_modes = len(modes)
num_dims = len(X)
num_regions = len(extents)

print(f"Number of extents = {num_regions}")

nn_bounds_dir = global_exp_dir + "/nn_bounds"
if not os.path.isdir(nn_bounds_dir):
    os.mkdir(nn_bounds_dir)

filename = global_exp_dir + f"/general_info"
use_personal = 1  # this means the corresponding output of the nn is the only input to the base kernel for each dim
np.save(filename, np.array([num_modes, num_dims, 1, num_regions, use_personal, individual_nns]))

use_cpu = True

tic = time.perf_counter()
lin_bounds = [[] for idx in range(num_regions)]  # using the same NN across all dimensions for one mode
linear_trans_m = [[] for idx in range(num_regions)]
linear_trans_b = [[] for idx in range(num_regions)]
for mode in modes:

    region_inf = region_info[mode]
    tic = time.perf_counter()
    if (os.path.exists(nn_bounds_dir + f"/linear_bounds_{mode + 1}_0.npy") or \
            os.path.exists(nn_bounds_dir + f"/linear_bounds_{mode + 1}_0_{num_dims}.npy")) and reuse_data:
        print(f"Skipping mode {mode + 1}, data is already saved.")
    else:
        if not individual_nns:
            # get NN bounds
            net = unknown_dyn_gp[mode][0][0].cpu().feature_extractor

            lin_bounds, linear_trans_m, linear_trans_b = bound_gelu_nn(extents, net, lin_bounds, linear_trans_m,
                                                                           linear_trans_b)

            filename = nn_bounds_dir + f"/linear_bounds_{mode + 1}_0"
            np.save(filename, np.array(lin_bounds))
            filename = nn_bounds_dir + f"/linear_trans_m_{mode + 1}_0"
            np.save(filename, np.array(linear_trans_m))
            filename = nn_bounds_dir + f"/linear_trans_b_{mode + 1}_0"
            np.save(filename, np.array(linear_trans_b))
        else:
            # need to bound nns for every dim and save appropriately
            for dim in range(num_dims):
                lin_bounds = [[] for _ in range(num_regions)]  # using the different NNs per dimension
                linear_trans_m = [[] for _ in range(num_regions)]
                linear_trans_b = [[] for _ in range(num_regions)]

                # get NN posteriors
                net = unknown_dyn_gp[mode][dim][0].cpu().feature_extractor
                
                lin_bounds, linear_trans_m, linear_trans_b = bound_gelu_nn(extents, net, lin_bounds,
                                                                               linear_trans_m, linear_trans_b)
                filename = nn_bounds_dir + f"/linear_bounds_{mode + 1}_0_{dim+1}"
                np.save(filename, np.array(lin_bounds))
                filename = nn_bounds_dir + f"/linear_trans_m_{mode + 1}_0_{dim+1}"
                np.save(filename, np.array(linear_trans_m))
                filename = nn_bounds_dir + f"/linear_trans_b_{mode + 1}_0_{dim+1}"
                np.save(filename, np.array(linear_trans_b))

    toc = time.perf_counter()
    print(f"Finished bounding the NN for mode {mode + 1} in {toc - tic} seconds")

    x_data = torch.tensor(np.transpose(region_inf[0]), dtype=torch.float32)
    y_data = np.transpose(region_inf[1])
    n_obs = max(np.shape(y_data))

    # identify which extents are in this region and their indices
    specific_extents = [j for j in range(0, num_regions)]
    c_vec = [0 for _ in range(num_dims)]
    B_vec = [0 for _ in range(num_dims)]

    for dim in range(num_dims):
        dim_region_filename = nn_bounds_dir + f"/linear_bounds_{mode + 1}_1_{dim + 1}"

        model = unknown_dyn_gp[mode][dim][0]
        model.cpu()  # unfortunately needs to be on cpu to access values

        nn_portion = model.feature_extractor
        with torch.no_grad():
            kernel_inputs = nn_portion.forward(x_data)
        if not individual_nns:
            kernel_inputs = torch.index_select(kernel_inputs, 1, torch.tensor(dim))

        noise_mat = pred_var[dim] * np.identity(np.shape(kernel_inputs)[0])

        covar_module = model.covar_module
        kernel_mat = covar_module(kernel_inputs)
        kernel_mat = kernel_mat.evaluate()
        K = kernel_mat.detach().numpy() + noise_mat
        # enforce symmetry, it is very close but causes errors when computing sig bounds
        K = (K + K.transpose()) / 2.
        K_inv = np.linalg.inv(K)  # only need to do this once per dim, yay

        Y_ = np.transpose(region_inf[1][dim, :])
        try:
            v = cp.Variable(kernel_data_points)
            constraints = [-process_dist["sig"][dim] <= v, v <= process_dist["sig"][dim]]
            objective = cp.Minimize(cp.quad_form(Y_ - v, K_inv))
            prob = cp.Problem(objective, constraints)
            check = prob.solve()
            if check > Bs[mode][dim] ** 2:
                c_opt = 0
            else:
                c_opt = check
        except:
            c_opt = 0

        c_vec[dim] = c_opt
        B_vec[dim] = Bs[mode][dim]

        y_dim = np.reshape(y_data[:, dim], (n_obs,))
        alpha_vec = K_inv @ y_dim

        lam_y_vec = np.sign(K_inv @ np.abs(y_dim))
        lam_x_vec = K_inv @ lam_y_vec.transpose()

        length_scale = model.covar_module.base_kernel.lengthscale.item()
        output_scale = model.covar_module.outputscale.item()

        # convert to julia input structure
        x_gp = np.array(np.transpose(kernel_inputs.detach().numpy())).astype(np.float64)
        K = np.array(K)
        K_inv = np.array(K_inv)
        alpha_vec = np.array(alpha_vec)
        lam_x_vec = np.array(lam_x_vec)
        out_2 = output_scale
        len_2 = length_scale ** 2.
        theta_vec, theta_vec_2 = theta_vectors(x_gp, len_2)

        # need to save x_gp, K, K_inv, alpha_vec, out_2, len_2, theta_vec, theta_vec_2, K_inv_s all individually
        np.save(dim_region_filename + "_x_gp", x_gp)
        np.save(dim_region_filename + "_theta_vec", theta_vec)
        np.save(dim_region_filename + "_theta_vec_2", theta_vec_2)
        np.save(dim_region_filename + "_K", K)
        np.save(dim_region_filename + "_K_inv", K_inv)
        np.save(dim_region_filename + "_alpha", alpha_vec)
        np.save(dim_region_filename + "_lam_x", lam_x_vec)
        np.save(dim_region_filename + "_kernel", np.array([out_2, len_2, ]))
        np.save(dim_region_filename + f"_these_indices_{0}", np.array(specific_extents))

    np.save(nn_bounds_dir + f"/c_opts_{mode}", np.array(c_vec))
    np.save(nn_bounds_dir + f"/B_info_{mode}", np.array(B_vec))
