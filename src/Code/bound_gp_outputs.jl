
@everywhere using PosteriorBounds
using PyCall
using SharedArrays
@everywhere include("quad_prog_bnb.jl")
@pyimport numpy

@everywhere function matching_label(test_label, compare_label)
    if isnothing(compare_label)
        return false
    end
    if test_label == "true"
        # any observation satisfies true
        return true
    end
    separated_test = Set(split(test_label, '∧'))
    separated_compare = Set(split(compare_label, '∧'))
    if issubset(separated_test, separated_compare)
        return true
    end
    if issubset(separated_compare, separated_test)
        return true
    end
    return false
end


function bound_gp_rkhs(num_regions, num_modes, num_dims, refinement, global_exp_dir, reuse_regions, use_single_dim,
                    dyn_noise, individual_nn; bound_thesis=false)
    nn_bounds_dir = global_exp_dir * "/nn_bounds"
    gp_bounds_dir = global_exp_dir * "/gp_bounds"

    if !isdir(gp_bounds_dir)
        mkdir(gp_bounds_dir)
    end
    for mode in 1:num_modes
        if isfile(gp_bounds_dir*"/complete_$mode" * "_$refinement.npy") && reuse_regions
            @info "Reusing prior bounds for mode $mode"
            continue
        end

        if refinement > 0
            mean_bound = numpy.load(gp_bounds_dir*"/mean_data_$mode" * "_$refinement.npy")
            mean_bound = convert(SharedArray, mean_bound)
            sig_bound = numpy.load(gp_bounds_dir*"/sig_data_$mode" * "_$refinement.npy")
            sig_bound = convert(SharedArray, sig_bound)
            additive_terms = numpy.load(gp_bounds_dir*"/additive_terms_$mode" * "_$refinement.npy")
            additive_terms = convert(SharedArray, sig_bound)
        else
            mean_bound = SharedArray(zeros(num_regions, num_dims, 2))
            sig_bound = SharedArray(zeros(num_regions, num_dims, 2))
            additive_terms = SharedArray(zeros(num_regions, num_dims, 2))
        end

        if individual_nn == 0
            linear_bounds = numpy.load(nn_bounds_dir*"/linear_bounds_$(mode)_$(refinement).npy")
            convert(SharedArray, linear_bounds)
        end

        mode_runtime = @elapsed begin
            for dim in 1:(num_dims::Int)
                dim_sig = dyn_noise[dim]
                # load all the data for this mode
                if individual_nn == 1
                    linear_bounds = numpy.load(nn_bounds_dir*"/linear_bounds_$(mode)_$(refinement)_$(dim).npy")
                    convert(SharedArray, linear_bounds)
                end

                dim_region_filename = nn_bounds_dir * "/linear_bounds_$(mode)_1_$dim"

                specific_extents = numpy.load(dim_region_filename*"_these_indices_$refinement.npy")  # need to add 1 to this
                x_gp = numpy.load(dim_region_filename*"_x_gp.npy")
                theta_vec = numpy.load(dim_region_filename*"_theta_vec.npy")
                theta_vec_2 = numpy.load(dim_region_filename*"_theta_vec_2.npy")
                K = numpy.load(dim_region_filename*"_K.npy")
                K_inv = numpy.load(dim_region_filename*"_K_inv.npy")
                alpha = numpy.load(dim_region_filename*"_alpha.npy")
                lam_x = numpy.load(dim_region_filename*"_lam_x.npy")
                kernel_info = numpy.load(dim_region_filename*"_kernel.npy")
                out2 = kernel_info[1]
                l2 = kernel_info[2]

                noise = K[1,1] - out2
                cK_inv_scaled = PosteriorBounds.scale_cK_inv(K, out2, noise)

                m = size(x_gp, 2) # n_obs
                n = size(x_gp, 1) # dims
                gp_lam = PosteriorBounds.PosteriorGP(n, m, x_gp, K, Matrix{Float64}(undef, m, m),
                        PosteriorBounds.UpperTriangular(zeros(m,m)), K_inv, -lam_x,
                        PosteriorBounds.SEKernel(out2, l2))  # - lam_x means a negative gp, hence minimizing is maximizing the normal gp
                gp_neg = PosteriorBounds.PosteriorGP(n, m, x_gp, K, Matrix{Float64}(undef, m, m),
                        PosteriorBounds.UpperTriangular(zeros(m,m)), K_inv, -alpha,
                        PosteriorBounds.SEKernel(out2, l2))
                gp = PosteriorBounds.PosteriorGP(n, m, x_gp, K, Matrix{Float64}(undef, m, m),
                        PosteriorBounds.UpperTriangular(zeros(m,m)), K_inv, alpha,
                        PosteriorBounds.SEKernel(out2, l2))
                PosteriorBounds.compute_factors!(gp)

                # parallelize getting mean bounds and variance bounds
                @sync @distributed for idx in specific_extents
                    # need distributed here because it is significant computation, threading is inefficient
                    if use_single_dim == 1 && individual_nn == 0
                        x_L = linear_bounds[idx+1, 1, dim]
                        x_U = linear_bounds[idx+1, 2, dim]
                        if x_U == x_L
                            x_U += 1e-5
                        end
                    else
                        x_L = linear_bounds[idx+1, 1, 1]
                        x_U = linear_bounds[idx+1, 2, 1]
                        if x_U == x_L
                            x_U += 1e-5
                        end
                    end

                    # get lower mean bounds
                    mean_info_l = PosteriorBounds.compute_μ_bounds_bnb(gp, x_L, x_U, theta_vec_2,
                                                                       theta_vec; max_iterations=100,
                                                                       bound_epsilon=1e-4, max_flag=false,
                                                                       prealloc=nothing)

                    # get upper mean bounds, negating alpha allows for the "min" to be the -max
                    mean_info_u = PosteriorBounds.compute_μ_bounds_bnb(gp_neg, x_L, x_U, theta_vec_2,
                                                                       theta_vec; max_iterations=100,
                                                                       bound_epsilon=1e-4, max_flag=false,
                                                                       prealloc=nothing)
                    mean_bound[idx+1, dim, 1] = mean_info_l[2]
                    mean_bound[idx+1, dim, 2] = -mean_info_u[2]


                    # find the x that maximizes the sum of Wx terms, this produces worst case bounds
                    mean_info_lam = PosteriorBounds.compute_μ_bounds_bnb(gp_lam, x_L, x_U, theta_vec_2,
                                                                       theta_vec; max_iterations=100,
                                                                       bound_epsilon=1e-4, max_flag=false,
                                                                       prealloc=nothing)
                    x_max_lam = mean_info_lam[1]*ones(size(x_gp))  # this is the x value that maximizes ||W_x||
                    K_xX = zeros(size(x_gp))
                    PosteriorBounds.cov!(K_xX, gp.kernel, x_max_lam, x_gp)
                    lam_x = 4*dim_sig*dim_sig*K_xX*K_inv*K_inv*K_xX'

                    det_bound_term = dim_sig*sum(abs.(K_xX*K_inv))  # this is sum sigma * [|W_x|]_i

                    additive_terms[idx+1, dim, 1] = lam_x[1]
                    additive_terms[idx+1, dim, 2] = det_bound_term

                    # get upper bounds on variance
                    close_req = 1e-4
                    sig_info = PosteriorBounds.compute_σ_bounds(gp, x_L, x_U, theta_vec_2, theta_vec,
                                                                cK_inv_scaled; max_iterations=50,
                                                                bound_epsilon=close_req, min_flag=false,
                                                                prealloc=nothing)

                    sig_low = sqrt(sig_info[3])  # this is a std deviation, actually an upper bound on the sigma
                    sig_upper = sqrt(sig_info[2])# this is a std deviation

                    if abs(sig_upper-sig_low) > sqrt(close_req)
                        # this means it didn't converge properly, use expensive quadratic program to find solution
                        outputs = sigma_bnb(gp, x_gp, m, n, out2, x_L, x_U, theta_vec, cK_inv_scaled;
                                            max_iterations=20, bound_epsilon=sqrt(close_req))
                        # @info "didn't converge" sig_upper sig_low outputs[3] outputs[2]
                        if abs(outputs[2]-outputs[3]) <= sqrt(close_req)
                            sig_upper = outputs[3]  # this is a std deviation
                        else
                            sig_upper = outputs[2]
                        end
                    else
                        if !isnan(sig_low)
                            sig_upper = sig_low
                        end
                    end

                    sig_bound[idx+1, dim, 1] = 0  # not actually needed for RKHS analysis
                    sig_bound[idx+1, dim, 2] = sig_upper
                end
            end
        end
        @info "Calculated bounds for mode $mode in $mode_runtime seconds"
        # save data
        numpy.save(gp_bounds_dir*"/mean_data_$mode" * "_$refinement", mean_bound)
        numpy.save(gp_bounds_dir*"/sig_data_$mode" * "_$refinement", sig_bound)

        numpy.save(gp_bounds_dir*"/additive_terms_$mode" * "_$refinement", additive_terms)
        numpy.save(gp_bounds_dir*"/complete_$mode" * "_$refinement", 1)

    end

end
