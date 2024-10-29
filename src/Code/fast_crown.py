
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


# this is on GPU and can't be parallelized, the cpu version probably can
def bound_gelu_nn(extents, net, linear_bounds_info, linear_transform_m, linear_transform_b):
    num_regions = len(extents)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    required_A = defaultdict(set)
    required_A[net.output_name[0]].add(net.input_name[0])

    for idx in range(num_regions):
        region_area = extents[idx]
        x_min = [k[0] for k in list(region_area)]
        x_max = [k[1] for k in list(region_area)]

        lower = torch.tensor([x_min])
        upper = torch.tensor([x_max])
        x = (upper - lower)/2.0 + lower
        ptb = PerturbationLpNorm(x_L=lower.to(device), x_U=upper.to(device))
        bounded_x = BoundedTensor(x.to(device), ptb)

        lb, ub, A = net.compute_bounds(x=(bounded_x,), method='alpha-CROWN', return_A=True, needed_A_dict=required_A)

        res = A[net.output_name[0]][net.input_name[0]]
        lA = res['lA'].cpu().detach().numpy()[0]
        uA = res['uA'].cpu().detach().numpy()[0]
        lbias = res['lbias'].cpu().detach().numpy()[0]
        ubias = res['ubias'].cpu().detach().numpy()[0]

        post_xmax = ub.cpu().detach().numpy()[0]
        post_xmin = lb.cpu().detach().numpy()[0]

        saved_vals = np.array([post_xmin.astype(np.float64), post_xmax.astype(np.float64)])
        linear_bounds_info[idx] = saved_vals
        saved_m = np.array([lA, uA])
        linear_transform_m[idx] = saved_m
        saved_b = np.array([lbias, ubias])
        linear_transform_b[idx] = saved_b

    return linear_bounds_info, linear_transform_m, linear_transform_b

