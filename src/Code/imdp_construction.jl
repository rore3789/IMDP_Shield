
using IntervalMDP
using Distributions
using SpecialFunctions
using SparseArrays
using Base.Threads
using Printf
using ProgressBars
using NetCDF
using MAT
using PyCall
@pyimport numpy


struct IMDPModel
    states
    actions
    Pmin
    Pmax
    labels
    extents
end


struct PIMDPModel
    states
    actions
    Pmin
    Pmax
    labels
    accepting_labels
    sink_labels
    extents
end


#======================================================================================================================#
#  Transition probability calculation
#======================================================================================================================#


function matching_label(test_label, compare_label)
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


function can_transition_check(num_dims::Int, mean_bounds, sig_bounds, add_terms, post, B_vec, c_opt_vec, noise_info,
                              delta, i, is_not_safe_set; prob=false, fixed_dim=nothing)
    if isnothing(fixed_dim)
        fixed_dim = [nothing for jj in 1:num_dims]
    end

    if is_not_safe_set
        can_transition = false
        for dim in 1:(num_dims::Int)
            # RKHS norm and c*
            c_opt = c_opt_vec[dim]
            B = B_vec[dim]

            # bounds on the mean of the image of the pre-region
            lower_mean = mean_bounds[i, dim, 1]
            upper_mean = mean_bounds[i, dim, 2]

            if !isnothing(fixed_dim[dim])
                if lower_mean < fixed_dim[dim][1]
                    lower_mean = fixed_dim[dim][1]
                    if upper_mean < fixed_dim[dim][1]
                        upper_mean = fixed_dim[dim][1] + 1e-4
                    end
                end
                if upper_mean > fixed_dim[dim][2]
                    upper_mean = fixed_dim[dim][2]
                    if lower_mean > fixed_dim[dim][2]
                        lower_mean = fixed_dim[dim][2] - 1e-4
                    end
                end
            end

            upper_sigma = sig_bounds[i, dim, 2] # this is a std deviation
            det_add_term = add_terms[i, dim, 2]
            lam_x = add_terms[i, dim, 1]

            a, b, var = noise_info[dim]

            minimum_bound = upper_sigma*sqrt(B*B - c_opt)
            if prob
                epsilon_region_det = minimum_bound + sqrt((lam_x/2)*log(2/delta)) + b
            else
                epsilon_region_det = minimum_bound + det_add_term + b  # b is the upper bound on noise, assumes symmetric zero mean
            end

            post_bounds = post[dim, :]
            post_low = post_bounds[1]
            post_up = post_bounds[2]

            lower = lower_mean - epsilon_region_det
            upper = upper_mean + epsilon_region_det
            if !isnothing(fixed_dim[dim])
                if lower < fixed_dim[dim][1]
                    lower = fixed_dim[dim][1]
                end
                if upper > fixed_dim[dim][2]
                    upper = fixed_dim[dim][2]
                end
            end
            stays_in = ((lower >= post_low) && (upper <= post_up))

            if !stays_in
                # the image does not intersect with the deterministic expansion of post
                can_transition = true
                break
            end
        end
    else
        can_transition = true
        for dim in 1:(num_dims::Int)
            # RKHS norm and c*
            c_opt = c_opt_vec[dim]
            B = B_vec[dim]

            # bounds on the mean of the image of the pre-region
            lower_mean = mean_bounds[i, dim, 1]
            upper_mean = mean_bounds[i, dim, 2]

            if !isnothing(fixed_dim[dim])
                if lower_mean < fixed_dim[dim][1]
                    lower_mean = fixed_dim[dim][1]
                    if upper_mean < fixed_dim[dim][1]
                        upper_mean = fixed_dim[dim][1] + 1e-4
                    end
                end
                if upper_mean > fixed_dim[dim][2]
                    upper_mean = fixed_dim[dim][2]
                    if lower_mean > fixed_dim[dim][2]
                        lower_mean = fixed_dim[dim][2] - 1e-4
                    end
                end
            end

            upper_sigma = sig_bounds[i, dim, 2] # this is a std deviation
            det_add_term = add_terms[i, dim, 2]
            lam_x = add_terms[i, dim, 1]

            a, b, var = noise_info[dim]

            minimum_bound = upper_sigma*sqrt(B*B - c_opt)
            if prob
                epsilon_region_det = minimum_bound + sqrt((lam_x/2)*log(2/delta)) + b
            else
                epsilon_region_det = minimum_bound + det_add_term + b  # b is the upper bound on noise, assumes symmetric zero mean
            end

            post_bounds = post[dim, :]
            post_low = post_bounds[1]
            post_up = post_bounds[2]

            lower = post_low - epsilon_region_det
            lower_in = ((lower_mean >= lower) && (lower_mean <= post_up + epsilon_region_det))
            upper_in = ((upper_mean <= post_up + epsilon_region_det) && (upper_mean >= lower))

            if !(lower_in || upper_in)
                # the mean prediction does not intersect with the deterministic expansion of post
                can_transition = false
                break
            end
        end
    end

    return can_transition
end


function get_interval_probs(num_dims::Int, mean_bounds, sig_bounds, add_terms, post, B_vec, c_opt_vec, noise_info,
                            cdf_fun, delta, i; prob=false, fixed_dim=nothing)
    if isnothing(fixed_dim)
        fixed_dim = [nothing for jj in 1:num_dims]
    end

    p_min = 1.0
    p_max = 1.0
    for dim in 1:(num_dims::Int)
        if p_max == 0.0
            continue
        end
        # RKHS norm and c*
        c_opt = c_opt_vec[dim]
        B = B_vec[dim]

        # bounds on the mean of the image of the pre-region
        lower_mean = mean_bounds[i, dim, 1]
        upper_mean = mean_bounds[i, dim, 2]

        if !isnothing(fixed_dim[dim])
            if lower_mean < fixed_dim[dim][1]
                lower_mean = fixed_dim[dim][1]
                if upper_mean < fixed_dim[dim][1]
                    upper_mean = fixed_dim[dim][1] + 1e-4
                end
            end
            if upper_mean > fixed_dim[dim][2]
                upper_mean = fixed_dim[dim][2]
                if lower_mean > fixed_dim[dim][2]
                    lower_mean = fixed_dim[dim][2] - 1e-4
                end
            end
        end

        a, b, var = noise_info[dim]

        upper_sigma = sig_bounds[i, dim, 2] # this is a std deviation
        lam_x = add_terms[i, dim, 1]
        det_add_term = add_terms[i, dim, 2]

        minimum_bound = upper_sigma*sqrt(B*B - c_opt)
        if prob
            epsilon_region_det = minimum_bound + sqrt((lam_x/2)*log(2/delta))
        else
            epsilon_region_det = minimum_bound + det_add_term
        end

        post_bounds = post[dim, :]
        post_low = post_bounds[1]
        post_up = post_bounds[2]

        lower = post_low - epsilon_region_det

        # check upper bound transition probs, find the noise such that there is NOT an
        # intersection with an expanded post
        lower_in = ((lower_mean >= lower) && (lower_mean <= post_up + epsilon_region_det))
        upper_in = ((upper_mean <= post_up + epsilon_region_det) && (upper_mean >= lower))

        upper_dist = upper_mean - (lower)
        lower_dist = (post_up + epsilon_region_det) - lower_mean

        if lower_in || upper_in
            # either the upper bound of the image has to be less than the min of the post
            # or the lower bound of the image has to be greater than the max of the post
            prob_mass = 1.0
            if upper_dist < b  && upper_dist > 0
                # can shift the image out of this post! will be a shift left, so negative noise
                x_noise = b - upper_dist
                # identify P(Noise < x_noise)
                prob_mass -= cdf_fun(-x_noise, a, b, var)
            end

            if lower_dist < b && lower_dist > 0
                # can shift the image out of this post! will be a shift right, so positive noise
                x_noise = b - lower_dist
                # identify P(Noise < x_noise)
                prob_mass -= (cdf_fun(x_noise, a, b, var) - cdf_fun(0, a, b, var))
            end
            p_max *= prob_mass

            # guaranteed to intersect for remaining noise probability
        else
            # find the distance to allow intersection and remove all other probability mass
            prob_mass = 0.0
            if upper_dist < 0  && abs(upper_dist) < b
                # the upper bound prediction is below the lower bound of post, find the shift that connects them
                x_noise = b + upper_dist  # how far it needs to move to the right connect
                prob_mass += cdf_fun(x_noise, a, b, var)
            end

            if lower_dist < 0  && abs(lower_dist) < b
                # the lower bound prediction is above the upper bound of post, find the shift that connects them
                x_noise = b + lower_dist  # how far it needs to move to the left connect
                prob_mass += cdf_fun(-x_noise, a, b, var)
            end

            p_max *= prob_mass
        end

        # check lower bound transition probs, find the noise s.t. the prediction is
        # entirely in the reduced post
        if p_min > 0

            addon = epsilon_region_det

            post_width = (post_up - addon) - (post_low + addon)  # this is the diameter of the shrunken post
            img_width = upper_mean - lower_mean  # diameter of image
            if (img_width <= post_width) && (post_width > 0)
                # the image has a possibility of fitting inside of the post

                upper_dist = (post_up - addon) - upper_mean # this needs to be positive
                lower_dist = lower_mean - (post_low + addon) # this also needs to be positive
                # if we're in here, they are either both positive or one is negative

                wiggle_room = min(upper_dist, lower_dist)
                if wiggle_room > 0
                    # already in post
                    # dists define how much the image can shift and still be in the post

                    up_shift = min(upper_dist, b)
                    down_shift = min(lower_dist, b)
                    p_min *= (cdf_fun(up_shift, a, b, var) - cdf_fun(-down_shift, a, b, var))
                else
                    if abs(wiggle_room) < b
                        # can shift to be in the post
                        if lower_dist < 0
                            # need to shift right, make sure upper bound stays in post as well
                            if abs(lower_dist) < upper_dist
                                max_shift = min(b, upper_dist)
                                p_min *= (cdf_fun(max_shift, a, b, var) - cdf_fun(-lower_dist, a, b, var))
                            else
                                # can't shift the lower end in without shifting the upper side out
                                p_min = 0
                            end
                        else
                            # need to shift left, make sure lower bound stays in post as well
                            if abs(upper_dist) < lower_dist
                                max_shift = min(b, lower_dist)
                                p_min *= (cdf_fun(upper_dist, a, b, var) - cdf_fun(-max_shift, a, b, var))
                            else
                                # can't shift the upper end in without shifting the lower side out
                                p_min = 0
                            end
                        end
                    else
                        # no chance to have the image entirely in post for any noise values
                        p_min = 0
                    end
                end

            else
                p_min = 0.0
            end
        end
    end

    return p_min, p_max
end


function Simple_RKHS_Synthesis_PIMDP(extents, noise_info, global_exp_dir, refinement, num_modes, num_regions, num_dims,
                                label_fn, dfa, imdp; noise_cdf="Uniform", use_prob=true, delta=0.05, fixed_dim=nothing)
    # this will use sparse arrays and the intervalIMDP.jl tool

    # define basic pimdp aspects from states and dfa
    dfa_states = sort(dfa["states"])
    states_sans_ends = copy(dfa_states)
    dfa_acc_state = dfa["accept"]
    dfa_sink_state = dfa["sink"]
    dfa_init_state = dfa["init"]

    sizeQ = length(dfa_states)
    addon = 0
    if !isnothing(dfa_acc_state)
        sizeQ -= 1
        addon += 1
        filter!(x -> x != dfa_acc_state, states_sans_ends)
    end

    if !isnothing(dfa_sink_state)
        sizeQ -= 1
        addon += 1
        filter!(x -> x != dfa_sink_state, states_sans_ends)
    end

    dfa_num_map = Dict()
    for i in 1:length(states_sans_ends)
        dfa_num_map[states_sans_ends[i]] = i
    end

    # Transition matrix size will be Nx|Q| -- or the original transition matrix permuted with the number of states in DFA
    N = (num_regions + 1) * num_modes
    M = num_regions + 1

    pimdp_states = []

    pimdp_actions = imdp.actions
    sizeA = length(pimdp_actions)

    for s in imdp.states
        for q in states_sans_ends
            new_state = (s, q)
            push!(pimdp_states, new_state)
        end
    end

    acc_labels = zeros(1, M*sizeQ + addon)
    if !isnothing(dfa_acc_state)
        acc_labels[M*sizeQ + 1] = 1
        push!(pimdp_states, (-1, dfa_acc_state))
    end

    sink_labels = zeros(1, M*sizeQ + addon)
    if !isnothing(dfa_sink_state)
        sink_labels[end] = 1
        push!(pimdp_states, (-1, dfa_sink_state))
    end

    # here is where we get transition probabilities from one state to another
    pbar = ProgressBar(total=(num_regions+1)*sizeQ*sizeA)

    # pre-load all bounds and store in new array
    all_means = Dict()
    all_sigs = Dict()
    all_adds = Dict()
    all_Bs = Dict()
    all_cs = Dict()
    nn_bounds_dir = global_exp_dir * "/nn_bounds"
    gp_bounds_dir = global_exp_dir  * "/gp_bounds"
    for mode in 1:(num_modes::Int)
        mean_bounds = numpy.load(gp_bounds_dir*"/mean_data_$mode" * "_$refinement.npy")
        sig_bounds = numpy.load(gp_bounds_dir*"/sig_data_$mode" * "_$refinement.npy")
        additive_terms = numpy.load(gp_bounds_dir*"/additive_terms_$mode" * "_$refinement.npy")
        all_means[mode] = mean_bounds
        all_sigs[mode] = sig_bounds
        all_adds[mode] = additive_terms

        c_vec = numpy.load(nn_bounds_dir * "/c_opts_$(mode-1).npy")
        b_vec = numpy.load(nn_bounds_dir * "/B_info_$(mode-1).npy")
        all_cs[mode] = c_vec
        all_Bs[mode] = b_vec
    end

    p_action_diff = []

    pimdp = PIMDPModel(pimdp_states, imdp.actions, nothing, nothing, imdp.labels, acc_labels, sink_labels,
                       imdp.extents)

    state_num = length(pimdp.states)
    action_num = length(pimdp.actions)

    # Get number of accepting states from the labels vector
    pimdp_acc_labels = pimdp.accepting_labels
    acc_states = findall(>(0.), pimdp_acc_labels[:])

    # note that the specification just becomes reachability on the product imdp
    reach_states = [acc_state for acc_state in acc_states]

    avg_transitions = 0

    is = Int[]
    js = Int[]
    low_vs = Float64[]
    upp_vs = Float64[]

    state_ptr = Int[]
    action_ptr = Int[]
    initial_states = Int[]

    if noise_cdf == "Uniform"
        cdf_fun = Uniform_CDF
    elseif noise_cdf == "Truncated_Normal"
        cdf_fun = Truncated_Normal_CDF
    elseif noise_cdf == "PERT"
        cdf_fun = PERT_CDF
    else
        @info "This noise distribution isn't supported yet"
        exit()
    end

    for i in 1:(num_regions::Int)
        for q in states_sans_ends
            qp_test = delta_(q, dfa, imdp.labels[i])  # which dfa state it transitions to
            p_action = 0
            num_trans = Atomic{Float64}(0)
            for mode in 1:(num_modes::Int)

                i_pimdp = (i-1)*(sizeQ * num_modes) + (dfa_num_map[q] - 1)*num_modes + mode  # this is the pimdp column number, includes action!

                if q == dfa_init_state && mode == 1
                    temp_val = (i-1)*sizeQ + dfa_num_map[q]
                    push!(initial_states, temp_val)
                end

                if mode == 1
                    push!(state_ptr, i_pimdp)
                end
                push!(action_ptr, mode)

                mean_bounds = all_means[mode]
                sig_bounds = all_sigs[mode]
                add_terms = all_adds[mode]
                c_opt_vec = all_cs[mode]
                B_vec = all_Bs[mode]
                sum_up = Atomic{Float64}(0)
                sum_low = Atomic{Float64}(0)

                using_sums = false
                if qp_test == dfa_acc_state
                    using_sums = true
                    j_pimdp = (M*sizeQ) + 1
                elseif qp_test == dfa_sink_state
                    using_sums = true
                    j_pimdp = (M*sizeQ) + addon
                end

                for j in 1:(num_regions+1::Int)
                    if using_sums == false
                        j_pimdp = (j-1)*sizeQ + dfa_num_map[qp_test]
                    end

                    post = extents[j, :, :]

                    # first check if there is a possible transition from deterministic upper bound
                    can_transition = can_transition_check(num_dims, mean_bounds, sig_bounds, add_terms, post, B_vec,
                                                          c_opt_vec, noise_info, delta, i, j==num_regions+1; prob=use_prob, fixed_dim=fixed_dim)

                    if can_transition
                        p_min, p_max = get_interval_probs(num_dims::Int, mean_bounds, sig_bounds, add_terms, post, B_vec,
                                                          c_opt_vec, noise_info, cdf_fun, delta, i; prob=use_prob, fixed_dim=fixed_dim)
                    else
                        p_min = 0.0
                        p_max = 0.0
                    end
                    p_min = round(p_min; digits=5)
                    p_max = round(p_max; digits=5)

                    if (j == num_regions + 1) && can_transition
                        p_min_ = deepcopy(p_min)
                        p_min = 1.0 - p_max
                        p_max = 1.0 - p_min_
                    end

                    atomic_add!(sum_up, p_max)
                    atomic_add!(sum_low, p_min)

                    if using_sums == false
                        if p_max > 0
                            atomic_add!(num_trans, 1.0)
                            push!(is, i_pimdp)
                            push!(js, j_pimdp)
                            push!(low_vs, p_min)
                            push!(upp_vs, p_max)
                        end
                    end
                end

                if using_sums == true
                    atomic_add!(num_trans, 1.0)
                    push!(is, i_pimdp)
                    push!(js, j_pimdp)
                    push!(low_vs, sum_low[])
                    push!(upp_vs, sum_up[])
                end

                if sum_up[] < 1.0
                    @info "upper bound is bad, $(sum_up[]), $i, $mode"
                    exit()
                end
                if sum_low[] > 1.0 || sum_low[] < 0.0
                    @info "lower bound is bad, $(sum_low[]), $i, $mode"
                    exit()
                end
                p_action += sum_up[] - sum_low[]
                ProgressBars.update(pbar)
            end
            avg_transitions += num_trans[]
            push!(p_action_diff, p_action)
        end
    end

    # self transitions outside of the defined space
    # do the same file writing for this state
    for q in states_sans_ends
        qp_test = delta_(q, dfa, imdp.labels[num_regions+1])  # which dfa state it transitions to
        for mode in 1:(num_modes::Int)
            i_pimdp = (num_regions)*(sizeQ * num_modes) + (dfa_num_map[q] - 1)*num_modes + mode  # this is the pimdp state number, includes action!
            if mode == 1
                push!(state_ptr, i_pimdp)
            end
            push!(action_ptr, mode)
            if qp_test == dfa_acc_state
                push!(is, i_pimdp)
                push!(js, (M*sizeQ) + 1)
                push!(low_vs, 1.0)
                push!(upp_vs, 1.0)
            elseif qp_test == dfa_sink_state
                push!(is, i_pimdp)
                push!(js, (M*sizeQ) + addon)
                push!(low_vs, 1.0)
                push!(upp_vs, 1.0)
            else
                qp_use = dfa_num_map[qp_test]
                col_idx = (num_regions)*sizeQ + qp_use
                push!(is, i_pimdp)
                push!(js, col_idx)
                push!(low_vs, 1.0)
                push!(upp_vs, 1.0)
            end
            ProgressBars.update(pbar)
        end
    end

    target_set = nothing
    if !isnothing(dfa_acc_state)
        i_pimdp = (M*(sizeQ * num_modes)) + 1
        push!(state_ptr, i_pimdp)
        push!(action_ptr, 1)
        push!(is, i_pimdp)
        push!(js, (M*sizeQ) + 1)
        push!(low_vs, 1.0)
        push!(upp_vs, 1.0)
        target_set = [(M*sizeQ) + 1]
    end

    avoid_set = nothing
    if !isnothing(dfa_sink_state)
        i_pimdp = (M*(sizeQ * num_modes)) + addon
        push!(state_ptr, i_pimdp)
        push!(action_ptr, 1)
        push!(is, i_pimdp)
        push!(js, (M*sizeQ) + addon)
        push!(low_vs, 1.0)
        push!(upp_vs, 1.0)
        avoid_set = [(M*sizeQ) + addon]
    end

    push!(state_ptr, state_ptr[end]+1)
    P_lower = sparse(js, is, low_vs)
    P_upper = sparse(js, is, upp_vs)

    js = nothing
    is = nothing
    low_vs = nothing
    upp_vs = nothing

#     validate_sparse_arrays(P_lower, P_upper)

    init_states = [1]  # this doesn't matter

    @info "This model has an average of $(avg_transitions/num_regions) transitions from each state"

    trans_probs = IntervalProbabilities(; lower = P_lower, upper = P_upper)
    mdp = IntervalMarkovDecisionProcess(trans_probs, state_ptr, action_ptr, init_states)


    pimdp = PIMDPModel(pimdp_states, imdp.actions, P_lower, P_upper, imdp.labels, acc_labels, sink_labels,
                       imdp.extents)

    return pimdp, mdp, p_action_diff, target_set, avoid_set, initial_states
end


#======================================================================================================================#
#  CDFs
#======================================================================================================================#

function Uniform_CDF(x, l, u, sigma)
    # sigma is a dummy variable, to keep uniformity with other distributions
    return (x - l)/(u - l)
end

function Phi(x)
    return (1/2)*(1 + erf(x/sqrt(2)))
end

function Truncated_Normal_CDF(x, l, u, sigma)
    # assumes 0 mean
    Z = Phi(u/sigma) - Phi(l/sigma)
    return (Phi(x/sigma) - Phi(l/sigma))/Z
end

function PERT_CDF(x, l, u, sigma)
    return  cdf(Beta(3,3), (x - l)/(u - l))
end


#======================================================================================================================#
#  PIMDP/IMDP construction
#======================================================================================================================#


function label_states(labels, extents, unsafe_label, num_dims, num_regions)
    state_labels = []
    for idx in 1:(num_regions+1::Int)
        extent = extents[idx, :, :]
        if idx == num_regions+1
            append!(state_labels, [unsafe_label])
            continue
        end
        possible_labels = []
        for label in keys(labels)
            # does this extent fit in any labels
            in_ranges = false
            ranges = labels[label]
            for sub_range in ranges
                in_sub_range = true
                if isnothing(sub_range)
                    in_ranges = false
                    break
                end
                for dim in 1:(num_dims::Int)

                    if !(extent[dim, 1] >= sub_range[dim][1] && extent[dim, 2] <= sub_range[dim][2])
                        in_sub_range = false
                        break
                    end
                end
                if in_sub_range
                    in_ranges = true
                    break
                end
            end

            if in_ranges
                append!(possible_labels, [label])
            else
                append!(possible_labels, ["!" * label])
            end
        end
        extent_label = ""
        for idx_ in 1:length(possible_labels)
            label = possible_labels[idx_]
            if idx_ > 1
                extent_label *= '∧' * label
            else
                extent_label *= label
            end
        end
        append!(state_labels, [extent_label])
    end

    return state_labels

end


function delta_(q, dfa, label)
    # returns the dfa state that can be transitioned to from q under the label
    trans = dfa["trans"]

    labels = []
    for relation in trans
        test_label = relation[2]
        output = relation[3]
        if relation[1] == dfa["accept"]
            # make sure you can't fail once you win?
            test_label = "true"
            output = relation[1]
        end
        if relation[1] == q && matches(test_label, label)
            return output
        end
    end
end


function matches(test_label, compare_label)
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


function find_q(q, labels)
    q_return = []
    for (idx, label) in enumerate(labels)
        if matches(label, q)
            append!(q_return, [idx])
        end
    end
    return q_return
end

