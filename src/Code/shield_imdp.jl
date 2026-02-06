

using Random
using Distributions
using IntervalMDP
using SparseArrays
using Base.Threads


function sys_2d_dynamics(support)
    # dynamics shouldn't really be here, but I didn't want another script just for one sample dynamical system
    mode1 = (x) -> [x[1] + 0.5 + 0.2*sin(x[2]),
                    x[2] + 0.4*cos(x[1])]

    mode2 = (x) -> [x[1] - 0.5 + 0.2*sin(x[2]),
                    x[2] + 0.4*cos(x[1])]

    mode3 = (x) -> [x[1] + 0.4*cos(x[2]),
                    x[2] + 0.5 + 0.2*sin(x[1])]

    mode4 = (x) -> [x[1] + 0.4*cos(x[2]),
                    x[2] - 0.5 + 0.2*sin(x[1])]

    process_noise = Uniform(-support, support)
    f1 = (x) -> mode1(x) + rand(process_noise, (2,1))
    f2 = (x) -> mode2(x) + rand(process_noise, (2,1))
    f3 = (x) -> mode3(x) + rand(process_noise, (2,1))
    f4 = (x) -> mode4(x) + rand(process_noise, (2,1))

    return [f1, f2, f3, f4]
end


function shield_algorithm(imdp, avoid_set, threshold, eps)
    """
        The function performs iterative value iteration on an IMDP to remove actions from stationary policies that reach
        an unsafe set. If a state has no action that can be safe, this retains the safest possible action so the shield
        never says "do nothing". Assess the values returned to see if the state is safe or is just retaining an action.

        Note that, at least in julia, the first call of this will be slightly slower as it needs to set up threads
        later calls should be faster

        Args:
            imdp: an imdp model defined with the IntervalMDP.jl package
                  transition_prob.lower: lower bound transition probabilities as a sparse array, with imdp state-action pairs as columns and states as rows
                  transition_prob.gap: the gap between lower and upper bound transition probabilities
                  action_vals: an array that represents the actions that are available for each IMDP state
                               this code assumes actions are defined as integers
                  num_states: the number of states in the IMDP
                  stateptr: for an IMDP state i stateptr[i] is the column index for the transition probabilities, i.e.
                            the lower bound transition prob from state i with the first action is
                            transition_prob.lower[stateptr[i]]
            avoid_set: the set of states in the IMDP the shield should prevent reaching, in a Product IMDP this would be
                       the states that are accepting for !(safety specification)
            threshold: the allowed violation probability for satisfying safety
            eps: the convergence threshold for terminating value iteration
        Returns:
            modes_avail: the shield, with an array of actions at each state in the IMDP
            Val_iter: the probability of violating safety under the worst remaining policy in the shield at each state
    """

    # first calculate the upper transition prob matrix, needing to look at gap and lower is much slower...
    trans_upper = imdp.transition_prob.lower + imdp.transition_prob.gap  # this is a sparse array
    trans_lower = imdp.transition_prob.lower

    # assign initial values of 0 everywhere and 1 in the regions to avoid
    Val_iter_init = zeros(Float32, imdp.num_states)
    Val_iter_init[avoid_set] .= 1.0f0
    Val_iter = copy(Val_iter_init)

    # create the initial shield by allowing all actions at every state and pre-compute a
    # dictionary (state, action) -> column index
    modes_avail = Vector{Vector{Int}}(undef, imdp.num_states)
    action_index = Vector{Dict{Int, Int}}(undef, imdp.num_states)
    for kk in 1:(imdp.num_states::Int)
        init_idx = imdp.stateptr[kk]
        end_idx = imdp.stateptr[kk+1] - 1
        modes_avail[kk] = copy(imdp.action_vals[init_idx:end_idx])

        dict = Dict{Int, Int}()
        for (offset, a) in enumerate(imdp.action_vals[init_idx:end_idx])
            dict[a] =  init_idx + offset - 1
        end
        action_index[kk] = dict
    end

    # now perform value iteration, removing actions and resetting as needed
    thread_shield_vals = [zeros(Float32, maximum(length.(modes_avail))) for _ in 1:Threads.nthreads()]
    reset = false
    Val_iter_new = copy(Val_iter)
    while true
        if reset
            Val_iter .= Val_iter_init
            reset = false
        end

        thread_max_diff = zeros(Float32, Threads.nthreads())
        thread_changed = falses(Threads.nthreads())

        Threads.@threads for ii in 1:(imdp.num_states::Int)
            tid = Threads.threadid()
            safe_actions = modes_avail[ii]  # current actions available in the shield
            n = length(safe_actions)
            new_safe_actions = Int[]
            shield_vals = thread_shield_vals[tid]  # will store values related to each action
            new_val = -9e9  # the probability under the least safe action

            @inbounds for a in 1:n
                # get the transition probs for this state action pair, this assumes sparse arrays so the next few lines
                # are used for efficiency, dense arrays would already have more efficient access to values but my need
                # more memory
                val_idx = action_index[ii][safe_actions[a]]

                lcol = trans_lower[:, val_idx]
                ucol = trans_upper[:, val_idx]
                rows = ucol.nzind  # transitions that have a non-zero upper bound probability

                lower_probs = Float32[get(lcol, r, 0.0f0) for r in rows]
                upper_probs = ucol.nzval  # fastest access to upper probs

                # sort transitions based on highest value first
                all_vals = @view Val_iter[rows]
                p = sortperm(all_vals; rev = true)
                vals_sorted = @view all_vals[p]
                lower_sorted = @view lower_probs[p]
                upper_sorted = @view upper_probs[p]

                # calculate the value for this action with an adversary that pushes transitions to be unsafe
                total_prob = Float32(sum(lower_sorted)) # ensure that we respect lower bound transition probs
                val = sum(lower_sorted .* vals_sorted)
                @inbounds for i in eachindex(vals_sorted)
                    # assign value based on lower probs first, then add extra based on gap to maximize value
                    remainder = 1.0f0 - total_prob  # how much probability is left in a valid distribution
                    added_prob = min(remainder, upper_sorted[i])
                    val += added_prob * vals_sorted[i]
                    total_prob += added_prob # can only every get to 1.0, break then
                    total_prob >= 1.0f0 && break
                end

                shield_vals[a] = val
                if val < Float32(threshold)
                    # if the value is below the violation threshold, this action is safe
                    push!(new_safe_actions, safe_actions[a])
                    new_val = max(new_val, val)
                end
            end

            if new_val >= 0
                # there is a safe action, use the least safe value
                Val_iter_new[ii] = new_val
            else
                # there is no safe action, retain the safest action and use that
                Val_iter_new[ii] = minimum(view(shield_vals, 1:n))
            end

            thread_max_diff[tid] = max(thread_max_diff[tid], abs(Val_iter_new[ii] - Val_iter[ii]))

            if safe_actions != new_safe_actions
                # shield changed, value iteration will need to restart after this
                if length(new_safe_actions) > 0
                    modes_avail[ii] = new_safe_actions
                    thread_changed[tid] = true
                else
                    # retain the safest action
                    modes_avail[ii] = [safe_actions[argmin(view(shield_vals, 1:n))]]
                    if length(safe_actions) > 1
                        thread_changed[tid] = true
                    end
                end
            end
        end

        if any(thread_changed)
            # shield changed the set of actions, restart
            reset = true
        else
            Val_iter .= Val_iter_new
            if maximum(thread_max_diff) < eps
                break  # converged
            end
        end
    end

    # return the shield (modes_avail) and the associated values under the worst remaining policy
    return modes_avail, Val_iter
end
