
from fast_crown import *


# ======================================================================
# 0. Example data generator for dynamics that can be sampled from
# ======================================================================

def g_2d_mode0(x):
    result = [x[0] + 0.5 + 0.2*np.sin(x[1]),
              x[1] + 0.4*np.cos(x[0])]
    return result


def g_2d_mode1(x):
    result = [x[0] + -0.5 + 0.2*np.sin(x[1]),
              x[1] + 0.4*np.cos(x[0])]
    return result


def g_2d_mode2(x):
    result = [x[0] + 0.4*np.cos(x[1]),
              x[1] + 0.5 + 0.2*np.sin(x[0])]
    return result


def g_2d_mode3(x):
    result = [x[0] + 0.4*np.cos(x[1]),
              x[1] + -0.5 + 0.2*np.sin(x[0])]
    return result


def generate_training_data(unknown_fnc, domain, data_num, known_fnc=None, random_seed=11, n_dims_out=-1,
                           process_dist=None, measurement_dist=None):
    # This function generates i.i.d. data points across the domain and adds noise from a specified distribution
    f_sub = unknown_fnc

    n_dims_in = len(domain)
    if n_dims_out > 0:
        n_dims_out = n_dims_out
    else:
        n_dims_out = n_dims_in

    np.random.seed(random_seed)

    keys = list(domain)
    x_train = [np.random.uniform(domain[k][0], domain[k][1], data_num) for k in keys]
    x_train = np.reshape(x_train, [n_dims_in, data_num])

    y_train = [f_sub([x_train[idx_i][idx_j] for idx_i in range(n_dims_in)]) for idx_j in range(data_num)]
    y_train = np.transpose(np.reshape(y_train, [data_num, n_dims_out]))

    if process_dist is not None:
        # this is essentially the same as measurement noise for this method of generating data
        sig = process_dist["sig"][0]
        if process_dist["dist"] == "Uniform":
            noise = np.random.uniform(low= -sig, high = sig, size=(n_dims_out, data_num))
        else:
            noise = np.random.multivariate_normal(process_dist["mu"], np.diag(sig), data_num).transpose()
        y_train += noise

    if measurement_dist is not None:
        # this is essentially the same as measurement noise for this method of generating data
        sig = process_dist["sig"][0]
        if "dist" in list(process_dist):
            noise = np.random.uniform(low= -sig, high = sig, size=(n_dims_out, data_num))
        else:
            noise = np.random.multivariate_normal(process_dist["mu"], np.diag(sig), data_num).transpose()
        y_train += noise

    if known_fnc is not None:
        y_train -= np.transpose(
            np.reshape([known_fnc([x_train[idx_i][idx_j] for idx_i in range(n_dims_in)]) for idx_j in range(data_num)],
                       [data_num, n_dims_out]))

    assert np.size(x_train) == np.size(y_train)

    return x_train, y_train


# ======================================================================
# 1. Define NN structure for AutoLirpa to bound outputs
# ======================================================================

class DynModelNetTanhExtreme(nn.Module):
    def __init__(self, d=1, width_1=1, width_2=1, out_dim=1):
        super().__init__()
        self.layer_1 = nn.Linear(d, width_1)
        self.layer_2 = nn.Linear(width_1, width_2)
        self.layer_3 = nn.Linear(width_2, width_2)
        self.layer_4 = nn.Linear(width_2, width_1)
        self.layer_5 = nn.Linear(width_1, width_2)
        self.layer_6 = nn.Linear(width_2, width_1)
        self.layer_7 = nn.Linear(width_1, out_dim)
        self.Tanh = nn.Tanh()

    def forward(self, x):
        out = self.layer_1(x)
        out = self.Tanh(out)
        out = self.layer_2(out)
        out = self.Tanh(out)
        out = self.layer_3(out)
        out = self.Tanh(out)
        out = self.layer_4(out)
        out = self.Tanh(out)
        out = self.layer_5(out)
        out = self.Tanh(out)
        out = self.layer_6(out)
        out = self.Tanh(out)
        out = self.layer_7(out)
        return out


class DynModelNetGelu4(nn.Module):
    def __init__(self, d=1, width_1=1, width_2=1, out_dim=1):
        super().__init__()
        self.layer_1 = nn.Linear(d, width_1)
        self.layer_2 = nn.Linear(width_1, width_2)
        self.layer_3 = nn.Linear(width_2, width_2)
        self.layer_4 = nn.Linear(width_2, width_1)
        self.layer_5 = nn.Linear(width_1, out_dim)
        self.GeLU = nn.GELU(approximate='tanh')

    def forward(self, x):
        out = self.layer_1(x)
        out = self.GeLU(out)
        out = self.layer_2(out)
        out = self.GeLU(out)
        out = self.layer_3(out)
        out = self.GeLU(out)
        out = self.layer_4(out)
        out = self.GeLU(out)
        out = self.layer_5(out)
        return out


class DynModelNetGelu5(nn.Module):
    def __init__(self, d=1, width_1=1, width_2=1, out_dim=1):
        super().__init__()
        self.layer_1 = nn.Linear(d, width_1)
        self.layer_2 = nn.Linear(width_1, width_2)
        self.layer_3 = nn.Linear(width_2, width_2)
        self.layer_4 = nn.Linear(width_2, width_2)
        self.layer_5 = nn.Linear(width_2, width_2)
        self.layer_6 = nn.Linear(width_2, out_dim)
        self.GeLU = nn.GELU(approximate='tanh')

    def forward(self, x):
        out = self.layer_1(x)
        out = self.GeLU(out)
        out = self.layer_2(out)
        out = self.GeLU(out)
        out = self.layer_3(out)
        out = self.GeLU(out)
        out = self.layer_4(out)
        out = self.GeLU(out)
        out = self.layer_5(out)
        out = self.GeLU(out)
        out = self.layer_6(out)
        return out


class DynModelNetTanh3(nn.Module):
    def __init__(self, d=1, width_1=1, width_2=1, out_dim=1):
        super().__init__()
        self.layer_1 = nn.Linear(d, width_1)
        self.layer_2 = nn.Linear(width_1, width_2)
        self.layer_3 = nn.Linear(width_2, width_2)
        self.layer_4 = nn.Linear(width_2, out_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.layer_1(x)
        out = self.tanh(out)
        out = self.layer_2(out)
        out = self.tanh(out)
        out = self.layer_3(out)
        out = self.tanh(out)
        out = self.layer_4(out)
        return out


class DynModelNetGelu2(nn.Module):
    def __init__(self, d=1, width_1=1, width_2=1, out_dim=1):
        super().__init__()
        self.layer_1 = nn.Linear(d, width_1)
        self.layer_2 = nn.Linear(width_1, width_2)
        self.layer_3 = nn.Linear(width_2, out_dim)
        self.GeLU = nn.GELU(approximate='tanh')

    def forward(self, x):
        out = self.layer_1(x)
        out = self.GeLU(out)
        out = self.layer_2(out)
        out = self.GeLU(out)
        out = self.layer_3(out)
        return out


class DynModelNetTanh2(nn.Module):
    def __init__(self, d=1, width_1=1, width_2=1, out_dim=1):
        super().__init__()
        self.layer_1 = nn.Linear(d, width_1)
        self.layer_2 = nn.Linear(width_1, width_2)
        self.layer_3 = nn.Linear(width_2, out_dim)
        self.tanh = nn.Tanh()
        self.GeLU = nn.GELU(approximate='tanh')

    def forward(self, x):
        out = self.layer_1(x)
        out = self.tanh(out)
        out = self.layer_2(out)
        out = self.tanh(out)
        out = self.layer_3(out)
        return out


class DynModelNetGelu1(nn.Module):
    def __init__(self, d=1, width_1=1, width_2=1, out_dim=1):
        super().__init__()
        self.layer_1 = nn.Linear(d, width_1)
        self.layer_2 = nn.Linear(width_1, out_dim)
        self.GeLU = nn.GELU(approximate='tanh')

    def forward(self, x):
        out = self.layer_1(x)
        out = self.GeLU(out)
        out = self.layer_2(out)
        return out


class DynModelNetTanh1(nn.Module):
    def __init__(self, d=1, width_1=1, width_2=1, out_dim=1):
        super().__init__()
        self.layer_1 = nn.Linear(d, width_1)
        self.layer_2 = nn.Linear(width_1, out_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.layer_1(x)
        out = self.tanh(out)
        out = self.layer_2(out)
        return out


class StandardGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(StandardGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        lengthscale_prior = gpytorch.priors.GammaPrior(1.0, 2.0)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(lengthscale_prior=lengthscale_prior))

    def forward(self, x):
        # not a deep kernel, this is a standard RBF kernel
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DeepKernelGPINN(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(DeepKernelGPINN, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        lengthscale_prior = gpytorch.priors.GammaPrior(1.0, 2.0)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(lengthscale_prior=lengthscale_prior))
        self.feature_extractor = feature_extractor

    def forward(self, x):
        # We're first putting our data through a deep net (feature extractor)
        # this assumes an individual feature extractor per dimension of the dynamics
        projected_x = self.feature_extractor(x) # will output 1 dimension
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DeepKernelGPPNN(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(DeepKernelGPPNN, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        lengthscale_prior = gpytorch.priors.GammaPrior(1.0, 2.0)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(lengthscale_prior=lengthscale_prior))
        self.feature_extractor = feature_extractor
        self.dim = torch.tensor(dimension_used).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def forward(self, x):
        # We're first putting our data through a deep net (feature extractor)
        # this has a single feature extractor for all dimensions, but only uses the relavant output dimension
        projected_x = self.feature_extractor(x)
        projected_x = torch.index_select(projected_x, 1, self.dim)  # take only the relevant dim
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# ======================================================================
#  2. Discretization Functions
# ======================================================================


def get_grid_info(X, num_regions):
    large_grid = {k: X[k][1] - X[k][0] for k in list(X)}
    grid_size = {k: large_grid[k]/num_regions[k] for k in list(X)}

    return grid_size, large_grid


def discretize_space_list(space, grid_size, include_space=True):
    # construct a finite discretization of space based on the number of partitions per dimension in grid_Size
    
    extents = []
    space_list = []
    for k in list(space):
        x = space[k]
        space_list.append(x)
        x_d = np.arange(x[0], x[1] + grid_size[k]/2., grid_size[k]).tolist()
        x_d = [round(x, 6) for x in x_d]
        extents.append([[x_d[i], x_d[i + 1]] for i in range(len(x_d) - 1)])

    state_extents = (itertools.product(*extents))
    discrete_sets = []
    for i, state in enumerate(state_extents):
        for j, k in enumerate(list(space)):
            if j == 0:
                discrete_sets.append([state[j]])
            else:
                discrete_sets[i].append(state[j])

    if include_space:
        discrete_sets.append(space_list)

    return discrete_sets


def discretize_space_defined(space, grid_info, include_space=True):
    extents = []
    space_list = []
    for k in list(space):
        x = space[k]
        space_list.append(x)
        x_d = grid_info[k]
        x_d = [round(x, 6) for x in x_d]
        extents.append([[x_d[i], x_d[i + 1]] for i in range(len(x_d) - 1)])

    state_extents = (itertools.product(*extents))
    discrete_sets = []
    for i, state in enumerate(state_extents):
        for j, k in enumerate(list(space)):
            if j == 0:
                discrete_sets.append([state[j]])
            else:
                discrete_sets[i].append(state[j])

    if include_space:
        discrete_sets.append(space_list)

    return discrete_sets


def manual_disc(space, dim_ranges, include_space=True):
    extents = []
    space_list = []
    for k in list(dim_ranges):
        x = space[k]
        space_list.append(x)
        x_d = dim_ranges[k]
        extents.append([[x_d[i], x_d[i + 1]] for i in range(len(x_d) - 1)])

    state_extents = (itertools.product(*extents))
    discrete_sets = []
    for i, state in enumerate(state_extents):
        for j, k in enumerate(list(dim_ranges)):
            if j == 0:
                discrete_sets.append([state[j]])
            else:
                discrete_sets[i].append(state[j])

    if include_space:
        discrete_sets.append(space_list)

    return discrete_sets


def add_region(extents, r, keys):
    new_extents = extents.copy()

    for ii in reversed(range(len(extents))):
        region_area = extents[ii]
        ranges = list(region_area)
        x_min = [k[0] for k in ranges]
        x_max = [k[1] for k in ranges]

        contains_0 = True
        for dim in range(len(ranges)):
            if not (x_min[dim] <= 0 <= x_max[dim]):
                contains_0 = False
                break

        if contains_0:
            # remove this region from the copy, and refine it so that the ball of radius r is taken out
            new_extents.pop(ii)
            grid_info = {}
            refined_region = {}
            for idx, k in enumerate(keys):
                refined_region[k] = region_area[idx]
                test = region_area[idx].copy()
                if test[0] < 0:
                    test.insert(1, -r)
                    if test[2] > 0:
                        test.insert(2, r)
                else:
                    test.insert(1, r)
                grid_info[k] = test
            new_additions = discretize_space_defined(refined_region, grid_info, include_space=False)

            for rr in new_additions:
                new_extents.append(rr)
    return new_extents

# ======================================================================
#  Learning functions
# ======================================================================


def loss_function_smooth(unknown_dyn_model, y_data, x_data, reduction='sum'):
    y_predict = unknown_dyn_model(x_data)
    loss_fn = torch.nn.MSELoss(reduction=reduction)
    loss = loss_fn(y_predict, y_data)

    return loss


def scaling_fnc(y_in, dx=4., dy=4., dz=1., dw=1.):

    M, N = np.shape(y_in)
    # scale each dimension, rather than the whole input
    y_out = torch.clone(y_in)
    for n in range(N):
        if n == 0:
            scale = dx
        elif n == 1:
            scale = dy
        elif n == 2:
            scale = dz
        else:
            scale = dw
        y_out[:, n] = y_in[:, n]/scale

    return y_out


def train_feature_extractor_i(all_data, mode, network_dims, elw_all, random_seed=20, des_loss=0, dim=0):
    update_each_step = False

    net_dims = network_dims[dim]
    d = net_dims[0]
    width_1 = net_dims[1]
    width_2 = net_dims[2]
    out_dim = 1
    layers = net_dims[4]

    elw = elw_all[dim]
    epochs = elw[0]
    lr = elw[1]
    wd = elw[2]

    if layers == 1:
        net_model_to_use = DynModelNetGelu1
    elif layers == 2:
        net_model_to_use = DynModelNetGelu2
    elif layers == 3:
        net_model_to_use = DynModelNetTanh3
    elif layers == 4:
        net_model_to_use = DynModelNetTanh2
    else:
        net_model_to_use = DynModelNetTanh1

    x_train = all_data[mode][0]
    y_train = all_data[mode][1]

    n_samples = int(np.shape(x_train)[1]/50.)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x_data = torch.tensor(np.transpose(x_train), dtype=torch.float32, requires_grad=True).to(device)
    y_input = y_train[dim, :]

    y_data = torch.tensor(np.reshape(y_input, (len(y_input), 1)), dtype=torch.float32).to(device)

    torch.manual_seed(random_seed)
    global feature_extractor
    feature_extractor = net_model_to_use(d=d, width_1=width_1, width_2=width_2, out_dim=out_dim)
    feature_extractor.to(device)

    optimizer = torch.optim.Adam(feature_extractor.parameters(), lr=lr, weight_decay=wd)
    print("Training Neural Network on generated data for mode {} and dim {}...".format(mode+1, dim))
    min_loss = 9e9
    prev_loss = 9e9
    for t in range(epochs):
        # Forward pass: compute predicted y by passing x to the model.
        # get minibatch data
        perm = torch.randperm(y_data.size(0))
        idx = perm[:n_samples]

        y_mini = y_data[idx]
        x_mini = x_data[idx]

        # Compute loss.
        loss = loss_function_smooth(feature_extractor, y_mini, x_mini)

        min_loss = min(min_loss, loss.item())

        optimizer.zero_grad()

        if update_each_step:
            loss.backward()
            optimizer.step()
        else:
            if loss.item() <= prev_loss:
                loss.backward()
                optimizer.step()

        if loss.item() <= des_loss:
            prev_loss = loss.item()
            break

        prev_loss = loss.item()

    # print([prev_loss, min_loss])

    feature_extractor.to(torch.device('cpu'))


def train_feature_extractor(all_data, mode, network_dims, elw, random_seed=20, using_gelu=True, use_scaling=False,
                            des_loss=None):
    update_each_step = False
    epochs = elw[0]
    lr = elw[1]
    wd = elw[2]

    if des_loss is None:
        des_loss = 0
    else:
        des_loss = des_loss[mode]

    # This is how layers from the yaml are used, you can add your own architecture if you follow the format of these
    if using_gelu:
        if network_dims[4] == 1:
            net_model_to_use = DynModelNetGelu1
        elif network_dims[4] == 2:
            net_model_to_use = DynModelNetGelu2
        elif network_dims[4] == 5:
            net_model_to_use = DynModelNetGelu5
        else:
            net_model_to_use = DynModelNetGelu4
    else:
        net_model_to_use = DynModelNetTanh2

    d = network_dims[0]
    width_1 = network_dims[1]
    width_2 = network_dims[2]
    out_dim = network_dims[3]

    x_train = all_data[mode][0]
    y_train = all_data[mode][1]

    n_samples = int(np.shape(x_train)[1]/50.)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x_data = torch.tensor(np.transpose(x_train), dtype=torch.float32, requires_grad=True).to(device)
    y_data = torch.tensor(np.transpose(y_train), dtype=torch.float32).to(device)

    scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1.0, 1.0)

    if use_scaling:
        if out_dim == 5:
            y_data = scaling_fnc(y_data, dx=10., dy=10., dz=10., dw=10.)
        if out_dim == 3:
            y_data = scaling_fnc(y_data, dx=10., dy=2.)
        else:
            y_data = scale_to_bounds(y_data)

    torch.manual_seed(random_seed)
    global feature_extractor
    feature_extractor = net_model_to_use(d=d, width_1=width_1, width_2=width_2, out_dim=out_dim)
    feature_extractor.to(device)

    optimizer = torch.optim.Adam(feature_extractor.parameters(), lr=lr, weight_decay=wd)
    print("Training Neural Network on generated data for mode {}...".format(mode+1))
    prev_loss = 9e9
    min_loss = 9e9
    for t in range(epochs):
        # Forward pass: compute predicted y by passing x to the model.
        # get minibatch data
        perm = torch.randperm(y_data.size(0))
        idx = perm[:n_samples]

        y_mini = y_data[idx]
        x_mini = x_data[idx]

        # Compute loss.
        loss = loss_function_smooth(feature_extractor, y_mini, x_mini)
        min_loss = min(min_loss, loss.item())

        optimizer.zero_grad()

        if update_each_step:
            loss.backward()
            optimizer.step()
        else:
            if loss.item() <= prev_loss:
                loss.backward()
                optimizer.step()

        if loss.item() <= des_loss:
            break
        prev_loss = loss.item()
    # print([min_loss, loss.item()])
    feature_extractor.to(torch.device('cpu'))


def deep_kernel_learning(all_data, mode, keys, network_dims, alphas, elw, training_iter=40, lr=0.01, random_seed=11,
                         kernel_data_points=1, using_gelu=False, use_standard=False,
                         use_scaling=False, individual_nns=False, des_loss=None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not use_standard and not individual_nns:
        train_feature_extractor(all_data, mode, network_dims, elw, random_seed=random_seed,
                                using_gelu=using_gelu, use_scaling=use_scaling, des_loss=des_loss)

    if des_loss is None:
        des_loss = [0 for _ in range(len(keys))]

    print('Optimizing GP parameters\n')

    x_train = all_data[mode][0]
    y_train = all_data[mode][1]

    # get a random set from the data up to kernel_data_points
    torch.manual_seed(mode*2 + 5)
    perm = torch.randperm(np.shape(y_train)[1])
    used_data_idx = perm[:kernel_data_points]

    x_train = x_train[:, used_data_idx]
    y_train = y_train[:, used_data_idx]

    region_data = [None, None]
    region_data[0] = x_train
    region_data[1] = y_train

    y_train = np.transpose(y_train)
    x_data = torch.tensor(np.transpose(x_train), dtype=torch.float32, requires_grad=True).to(device)
    n_obs = np.shape(y_train)[0]

    gp_by_dim = []
    for dim, key in enumerate(keys):
        y_data = torch.tensor(np.reshape(y_train[:, dim], (n_obs,)), dtype=torch.float32).to(device)

        alpha_ = alphas[dim]
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        if not use_standard:
            if not individual_nns:
                global dimension_used
                dimension_used = dim
                model = DeepKernelGPPNN(x_data, y_data, likelihood)
            else:
                train_feature_extractor_i(all_data, mode, network_dims, elw, random_seed=random_seed, dim=dim,
                                          des_loss=des_loss[dim])
                model = DeepKernelGPINN(x_data, y_data, likelihood)

        else:
            model = StandardGP(x_data, y_data, likelihood)

        if torch.cuda.is_available():
            model = model.cuda()
            likelihood = likelihood.cuda()

        hypers = {'likelihood.noise_covar.noise': torch.tensor(alpha_), }

        model.initialize(**hypers)

        model.train()
        likelihood.train()

        # NOTE, this does not optimize the NN feature extractor, uses the pre-trained version
        optimizer = torch.optim.Adam([{'params': model.covar_module.parameters()},
                                      {'params': model.mean_module.parameters()},
                                      {'params': model.likelihood.parameters()},
                                      ], lr=lr)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(x_data)

            # Calc loss and backprop gradients
            loss = -mll(output, y_data)

            loss.backward()
            if i % training_iter == training_iter-1:
                print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   outputscale: %.3f' %
                      (i + 1, training_iter, loss.item(),
                       model.covar_module.base_kernel.lengthscale.item(),
                       model.covar_module.outputscale.item()))

            optimizer.step()

        model.eval()
        likelihood.eval()
        print("")
        gp_by_dim.append([model, likelihood])

    return gp_by_dim, region_data


# ======================================================================
#  Save and load data functions
# ======================================================================


def dict_save(file_name, dict_to_save):
    file_ = open(file_name, "wb")
    # write the python object (dict) to pickle file
    pickle.dump(dict_to_save, file_)
    # close file
    file_.close()


def dict_load(file_name):
    file_ = open(file_name, "rb")
    data = pickle.load(file_)
    file_.close()
    return data
