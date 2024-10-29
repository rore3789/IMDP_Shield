

using Random
using Distributions


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


function value_iter_shield(imdp, actions, avoid_set, threshold, time_horizon)
    # this version for any state that doesn't have an action with a high enough probability of safety will only allow
    # the safest possible action and will set the value accordingly

    Val_iter = zeros(imdp.num_states)
    for kk in avoid_set
        Val_iter[kk] = 1.0
    end

    modes_avail = []
    for kk in 1:(imdp.num_states::Int)
        init_idx = imdp.stateptr[kk]
        end_idx = imdp.stateptr[kk+1] - 1
        actions_ = [a for a in imdp.action_vals[init_idx:end_idx]]
        push!(modes_avail, actions_)
    end
    @info "Got through Modes Avail"

    # now perform value iteration, removing actions and resetting as needed
    reset = false
    counter = 0
    while true
        if (counter >= time_horizon) && time_horizon > 0
            break
        end

        if reset
            Val_iter = zeros(imdp.num_states)
            for kk in avoid_set
                Val_iter[kk] = 1.0
            end
            reset = false
        end

        Val_iter_new = deepcopy(Val_iter)

        changed_actions = false
        max_diff = 0
        for ii in 1:(imdp.num_states::Int)
            init_idx = imdp.stateptr[ii]
            safe_actions = modes_avail[ii]
            new_safe_actions = []
            shield_vals = []
            new_val = -9e9

            for a in safe_actions
                # get the transition probs for this state action pair

                new_idx = findfirst(==(a), imdp.action_vals[init_idx:end])
                val_idx = init_idx + new_idx - 1  # this is the column of the imdp probabilities that matches this state-action

                lower_probs = imdp.transition_prob.lower[:, val_idx]

                upper_probs = lower_probs + imdp.transition_prob.gap[:, val_idx]
                has_value = findall(x -> x>0, upper_probs)

                all_values = Val_iter[has_value]
                p = sortperm(all_values, rev=true)  # sort based on highest value first

                all_values = all_values[p]
                has_value = has_value[p]

                total_prob = Float32(sum(lower_probs))
                val = 0.0
                idx = 1
                for jj in has_value
                    # assign value based on lower probs first, then add extra based on gap to maximize value
                    remainder = 1.0 - total_prob
                    added_prob = min(remainder, upper_probs[jj])
                    val += (lower_probs[jj] + added_prob) * all_values[idx]
                    total_prob += added_prob
                    idx += 1
                end

                push!(shield_vals, val)
                if val < Float32(threshold)
                    push!(new_safe_actions, a)
                    new_val = max(new_val, val)
                end
            end

            if new_val >= 0
                Val_iter_new[ii] = new_val
            else
                Val_iter_new[ii] = minimum(shield_vals)
            end

            max_diff = max(max_diff, abs(Val_iter_new[ii] - Val_iter[ii]))

            if safe_actions != new_safe_actions
                if length(new_safe_actions) > 0
                    modes_avail[ii] = new_safe_actions
                    changed_actions = true
                else
                    modes_avail[ii] = [safe_actions[argmin(shield_vals)]]
                    if length(safe_actions) > 1
                        changed_actions = true
                    end
                end
            end

        end

        if changed_actions
            reset = true
            counter = 0
        else
            counter = counter + 1
            Val_iter = deepcopy(Val_iter_new)
            if max_diff < 1e-6
                break  # converged
            end
        end
    end

    return modes_avail, Val_iter
end

