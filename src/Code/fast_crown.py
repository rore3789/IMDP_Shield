
from import_script import *


def get_gp_bound_info(extents, nn_out_dim, linear_bounds_info, linear_transform_m, linear_transform_b):
    transform_ = np.eye(nn_out_dim)
    transform_ = transform_.reshape(1, nn_out_dim, nn_out_dim)

    bias_ = np.zeros([1, nn_out_dim])
    for index, region in enumerate(extents):
        x_min = [k[0] for k in list(region)]
        x_max = [k[1] for k in list(region)]
        saved_vals = [np.array(x_min).astype(np.float64), np.array(x_max).astype(np.float64)]
        linear_bounds_info[index] = saved_vals

        saved_m = np.array([transform_, transform_])
        linear_transform_m[index] = saved_m

        saved_b = np.array([bias_, bias_])
        linear_transform_b[index] = saved_b

    return linear_bounds_info, linear_transform_m, linear_transform_b


# TODO, can this be parallelized?
def bound_gelu_nn_cpu(extents, net, linear_bounds_info, linear_transform_m, linear_transform_b, check=None, dim=None):
    num_regions = len(extents)

    required_A = defaultdict(set)
    required_A[net.output_name[0]].add(net.input_name[0])

    if check is None:
        for idx in range(num_regions):
            region_area = extents[idx]
            x_min = [k[0] for k in list(region_area)]
            x_max = [k[1] for k in list(region_area)]

            if dim == 1:
                x_min = x_min[1:4]
                x_max = x_max[1:4]

            lower = torch.tensor([x_min])
            upper = torch.tensor([x_max])
            x = (upper - lower)/2.0 + lower
            ptb = PerturbationLpNorm(x_L=lower, x_U=upper)
            bounded_x = BoundedTensor(x, ptb)

            lb, ub, A = net.compute_bounds(x=(bounded_x,), method='alpha-CROWN', return_A=True, needed_A_dict=required_A)

            res = A[net.output_name[0]][net.input_name[0]]
            lA = res['lA'].detach().numpy()[0]
            uA = res['uA'].detach().numpy()[0]
            lbias = res['lbias'].detach().numpy()[0]
            ubias = res['ubias'].detach().numpy()[0]

            post_xmax = ub.detach().numpy()[0]
            post_xmin = lb.detach().numpy()[0]

            saved_vals = np.array([post_xmin.astype(np.float64), post_xmax.astype(np.float64)])
            linear_bounds_info[idx] = saved_vals
            saved_m = np.array([lA, uA])
            linear_transform_m[idx] = saved_m
            saved_b = np.array([lbias, ubias])
            linear_transform_b[idx] = saved_b

    else:
        region_area = extents[check]
        x_min = [k[0] for k in list(region_area)]
        x_max = [k[1] for k in list(region_area)]

        lower = torch.tensor([x_min])
        upper = torch.tensor([x_max])
        x = (upper - lower) / 2.0 + lower
        ptb = PerturbationLpNorm(x_L=lower, x_U=upper)
        bounded_x = BoundedTensor(x, ptb)

        lb, ub, A = net.compute_bounds(x=(bounded_x,), method='alpha-CROWN', return_A=True, needed_A_dict=required_A)

        res = A[net.output_name[0]][net.input_name[0]]
        lA = res['lA'].detach().numpy()[0]
        uA = res['uA'].detach().numpy()[0]
        lbias = res['lbias'].detach().numpy()[0]
        ubias = res['ubias'].detach().numpy()[0]

        post_xmax = ub.detach().numpy()[0]
        post_xmin = lb.detach().numpy()[0]

        print(f"{[post_xmin, post_xmax]}\n")

    return linear_bounds_info, linear_transform_m, linear_transform_b

# Batch call on GPU/CPU is much faster than any parallelized call on CPU, cpu version kept for comparison
def bound_gelu_nn(extents, net, linear_bounds_info, linear_transform_m, linear_transform_b, batch_size=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    x_ex = torch.tensor([[1. for _ in range(latent_dim + 1)]]).to(device)
    default_bound_opts = {
        'conv_mode': 'patches',
        'sparse_intermediate_bounds': False,
        'sparse_conv_intermediate_bounds': False,
        'sparse_intermediate_bounds_with_ibp': True,
        'sparse_features_alpha': True,
        'sparse_spec_alpha': True,
        'minimum_sparsity': 0.9,
        'enable_opt_interm_bounds': False,
        'crown_batch_size': np.inf,
        'forward_refinement': True,
        'dynamic_forward': True,
        'forward_max_dim': int(1e9),
        'use_full_conv_alpha': True,
        'disabled_optimization': [],
        'use_full_conv_alpha_thresh': 512,
        'verbosity': 0,
        'optimize_graph': {'optimizer': None},
        'enable_beta_crown': True,
        'enable_alpha_crown': True,
        'fix_interm_bounds': True,
    }
    network = BoundedModule(net, torch.empty_like(x_ex), device=torch.device(device), bound_opts=default_bound_opts)

    required_A = defaultdict(set)
    required_A[network.output_name[0]].add(network.input_name[0])
    original_stdout = sys.stdout
    with torch.no_grad(), open(os.devnull, 'w') as f:
        sys.stdout = f
        num_regions = len(extents)
        for i in tqdm.tqdm(range(0, num_regions, batch_size), desc="Computing Bounds", unit="batch"):
            end_idx = min(i+batch_size, num_regions)
            xs_vec = []
            lower_vec = []
            upper_vec = []
            for idx in range(i, end_idx):
                region = extents[idx]

                x_min = [region[dim][0] for dim in range(latent_dim)]

                x_max = [region[dim][1] for dim in range(latent_dim)]

                lower = torch.tensor([x_min], dtype=torch.float32)
                upper = torch.tensor([x_max], dtype=torch.float32)
                x = (upper - lower) / 2.0 + lower

                xs_vec.append(torch.reshape(x, (latent_dim + 1,)))
                lower_vec.append(torch.reshape(lower, (latent_dim + 1,)))
                upper_vec.append(torch.reshape(upper, (latent_dim + 1,)))

            xs = torch.stack(xs_vec).to(device)
            lowers = torch.stack(lower_vec).to(device)
            uppers = torch.stack(upper_vec).to(device)

            ps = PerturbationLpNorm(x_L=lowers, x_U=uppers)
            lbs, ubs, As = network.compute_bounds(x=(BoundedTensor(xs, ps),), method='alpha-CROWN', return_A=True, needed_A_dict=required_A)

            outputs = As[network.output_name[0]][network.input_name[0]]
            lA_ = outputs['lA'].cpu().detach().numpy()
            uA_ = outputs['uA'].cpu().detach().numpy()
            lbias_ = outputs['lbias'].cpu().detach().numpy()
            ubias_ = outputs['ubias'].cpu().detach().numpy()

            for index in range(lbs.shape[0]):
                post_xmax = lbs[index].cpu().detach().numpy()
                post_xmin = ubs[index].cpu().detach().numpy()

                saved_vals = np.array([post_xmin.astype(np.float64), post_xmax.astype(np.float64)])

                linear_bounds_info[index + i] = saved_vals

                saved_m = np.array([lA_[index], uA_[index]])
                linear_transform_m[index + i] = saved_m

                saved_b = np.array([lbias_[index], ubias_[index]])
                linear_transform_b[index + i] = saved_b

    sys.stdout = original_stdout
    return linear_bounds_info, linear_transform_m, linear_transform_b


