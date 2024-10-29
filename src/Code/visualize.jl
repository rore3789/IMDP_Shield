
using Plots
using ColorSchemes
using Printf
using Flux
using NPZ


function plot_shield_actions(shield, total_actions, extents, num_regions, num_dims, plot_dir, refinement, pimdp,
                             time_horizon, values, threshold, init_states, policy; use_prob=true, labeled_regions=Dict(), obs_key=nothing,
                             state_outlines_flag=false, num_dfa_states=1, modes=nothing, num_start_regions=1,
                             simulate=true, dfa=nothing, dfa_policy=nothing)
    X = extents[num_regions+1,:,:]
    minx = X[1, 1]
    maxx = X[1, 2]
    miny = X[2, 1]
    maxy = X[2, 2]

    plt_verification = plot(aspect_ratio=1,
                            size=(300,300), dpi=300,
                            xlims=[minx, maxx], ylims=[miny, maxy],
                            xtickfont=font(10),
                            ytickfont=font(10),
                            titlefont=font(10),
                            xticks = [minx, 0, maxx],
                            yticks = [miny, 0, maxy],
                            grid=false,
                            backgroundcolor=128)

    # Plot the cells
    plot_cell_verify(extents[end, :, :], 0.0, 0.0, .95, state_outlines_flag=state_outlines_flag)  # just created the domain

    safe_regions = []
    for ii in 1:num_regions
        num_safe = length(shield[init_states[ii]])
        if values[init_states[ii]] > threshold
            num_safe = 0  # this is for shield_mkII
        end

        plot_cell_shield(extents[ii, :, :], num_safe, total_actions)
        if num_safe > 0
            push!(safe_regions, ii)
        end
    end

    # fill obstacles with black
    if obs_key != nothing
        obs = labeled_regions[obs_key]
        for region in obs
            if isnothing(region)
                continue
            end
            plot_obs(region)
        end
    end

    # Plot extents with labels
    for key in keys(labeled_regions)
        one_label = labeled_regions[key]
        for region in one_label
            if isnothing(region)
                continue
            end
            color = :black
            if key == obs_key
                color = :white
            end
            plot_labelled_extent_outline(region, key; color=color)
        end
    end

    if use_prob
        savefig(plt_verification, plot_dir * "/shield_prob_results_$(refinement)_$(time_horizon).pdf")
    else
        savefig(plt_verification, plot_dir * "/shield_det_results_$(refinement)_$(time_horizon).pdf")
    end

    # now simulate a ton of times to verify
    if !isnothing(modes) && simulate
        @info "Simulating trajectories under random safe acitons..."

        dfa_states = sort(dfa["states"])
        states_sans_ends = copy(dfa_states)
        dfa_acc_state = dfa["accept"]
        dfa_sink_state = dfa["sink"]
        dfa_init_state = dfa["init"]

        sizeQ = length(dfa_states)
        if !isnothing(dfa_acc_state)
            sizeQ -= 1
            filter!(x -> x != dfa_acc_state, states_sans_ends)
        end
        if !isnothing(dfa_sink_state)
            sizeQ -= 1
            filter!(x -> x != dfa_sink_state, states_sans_ends)
        end
        dfa_num_map = Dict()
        for i in 1:length(states_sans_ends)
            dfa_num_map[states_sans_ends[i]] = i
        end

        max_steps = time_horizon>0 ? time_horizon : 1000

        avg_success_rate = 0
        num_initializations = 1000
        for kk in 1:num_start_regions
            success = 0.0

            # pick random region from regions with safe actions
            s = rand(safe_regions)

            region = extents[s, :, :]

            # simulate 1000 times from this region and find success rate
            q_init = delta_(dfa_init_state, dfa, pimdp.labels[s])
            q = q_init
            for _ in 1:num_initializations
                x = [region[d,1] + (region[d,2] - region[d,1])*rand() for d in 1:num_dims] # get an initial state from that region
                s = which_extent(extents, x, num_regions, num_dims)
                q = copy(q_init)
                traj_done = false
                steps = 1
                while !traj_done
                    row_idx = (s-1)*sizeQ + (dfa_num_map[q])
                    avail_actions = shield[row_idx]
                    if length(avail_actions) == 0
                        @info "Ended in a state with no actions"
                        break
                    end

                    action = floor(Int, rand(avail_actions))

                    f = modes[action]
                    x = f(x)

                    s = which_extent(extents, x, num_regions, num_dims)
                    if s == num_regions+1
                        @info "Trajectory left the domain: $x"
                        break
                    end
                    q = delta_(q, dfa, pimdp.labels[s])

                    if q == dfa_sink_state
                        traj_done = true
                    elseif steps >= max_steps
                        traj_done = true
                        success += 1.0
                    else
                        steps += 1
                    end
                end
            end
            avg_success_rate += (success/num_initializations)*100.0  # % success rate
        end
        @info "Simulated success rate over $(num_start_regions) initial regions with $(num_initializations) initializations per region is $(avg_success_rate/num_start_regions)"
    end

    if !isnothing(modes) && !isnothing(policy)
        @info "Simulating shielded trajectories under the policy"

        dfa_states = sort(dfa["states"])
        states_sans_ends = copy(dfa_states)
        dfa_acc_state = dfa["accept"]
        dfa_sink_state = dfa["sink"]
        dfa_init_state = dfa["init"]

        sizeQ = length(dfa_states)
        if !isnothing(dfa_acc_state)
            sizeQ -= 1
            filter!(x -> x != dfa_acc_state, states_sans_ends)
        end

        if !isnothing(dfa_sink_state)
            sizeQ -= 1
            filter!(x -> x != dfa_sink_state, states_sans_ends)
        end

        dfa_num_map = Dict()
        for i in 1:length(states_sans_ends)
            dfa_num_map[states_sans_ends[i]] = i
        end


        dfa_states_p = sort(dfa_policy["states"])
        states_sans_ends_p = copy(dfa_states_p)
        dfa_acc_state_p = dfa_policy["accept"]
        dfa_sink_state_p = dfa_policy["sink"]
        dfa_init_state_p = dfa_policy["init"]

        sizeQp = length(dfa_states_p)
        if !isnothing(dfa_acc_state_p)
            sizeQp -= 1
            filter!(x -> x != dfa_acc_state_p, states_sans_ends_p)
        end

        if !isnothing(dfa_sink_state_p)
            sizeQp -= 1
            filter!(x -> x != dfa_sink_state_p, states_sans_ends_p)
        end

        dfa_num_map_p = Dict()
        for i in 1:length(states_sans_ends_p)
            dfa_num_map_p[states_sans_ends_p[i]] = i
        end

        max_steps = 1000

        avg_completes_task = 0
        avg_shielded = 0
        avg_num_steps = 0
        success = 0.0
        num_tests = 1000
        traj_count = 0
        for kk in 1:num_tests
            s = rand(safe_regions)

            region = extents[s, :, :]

            # simulate 1000 times from this region and find success rate
            q_init = delta_(dfa_init_state, dfa, pimdp.labels[s])
            q = q_init

            q_init_p = delta_(dfa_init_state_p, dfa_policy, pimdp.labels[s])
            qp = q_init_p

            region = extents[s, :, :]
            x = [region[d,1] + (region[d,2] - region[d,1])*rand() for d in 1:num_dims]

            # simulate 1000 time steps validate safety rate
            completes_task = 0

            traj_done = false
            steps = 1
            shielded = 0
            x_vec = [x[1]]
            y_vec = [x[2]]

            shielding_steps = []
            while !traj_done
                state = Float32[x[1], x[2], dfa_num_map_p[qp]-1]
                action = argmax(policy(transpose(hcat(state...))))[1]

                row_idx = (s-1)*sizeQ + (dfa_num_map[q])
                avail_actions = shield[row_idx]

                if length(avail_actions) == 0
                    @info "Ended in a state with no actions"
                    avg_num_steps += steps
                    break
                end

                uses_shield = false
                if !(action in avail_actions)
                    uses_shield = true
                    shielded += 1
                    action = floor(Int, rand(avail_actions))
#                     action = avail_actions[1]
                end
                append!(shielding_steps, uses_shield)


                # transition dfa states and actual state
                q = delta_(q, dfa, pimdp.labels[s])
                qp = delta_(qp, dfa_policy, pimdp.labels[s])

                f = modes[action]
                x = f(x)

                append!(x_vec, [x[1]])
                append!(y_vec, [x[2]])

                s = which_extent(extents, x, num_regions, num_dims)
                if s == num_regions+1
                    @info "Trajectory left the domain: $x"
                    break
                end

                if qp == dfa_acc_state_p
                    completes_task += 1.0
                    qp = dfa_init_state_p
                end

                if steps > max_steps
                    traj_done = true
                    success += 1
                    avg_num_steps += steps
                elseif q == dfa_sink_state
                    traj_done = true
                    avg_num_steps += steps
                else
                    steps += 1
                end

            end

            avg_completes_task += completes_task
            avg_shielded += shielded
        end

        @info "Simulated success rate for remaining safe is $(success/num_tests * 100.0)"
        @info "Average number of shield interventions over an average of $(avg_num_steps/num_tests) steps is $(avg_shielded/num_tests)"
        @info "Average number of times the task is completed over an average of $(avg_num_steps/num_tests) steps is $(avg_completes_task/num_tests)"

    end



end

function plot_obs(extent; color=:black, fill=1.)
    x = [extent[1][1], extent[1][1], extent[1][2], extent[1][2]]
    y = [extent[2][1], extent[2][2], extent[2][2], extent[2][1]]
    shape = Plots.Shape(x, y)
    plot!(shape, color=color, fillalpha=fill, linecolor=:black, linewidth=1, label="")
end


function plot_labelled_extent_outline(extent, label; color=:white)
    x = [extent[1][1], extent[1][1], extent[1][2], extent[1][2]]
    y = [extent[2][1], extent[2][2], extent[2][2], extent[2][1]]
    shape = Plots.Shape(x, y)
    plot!(shape, fillalpha=0, linecolor=:black, linewidth=1, label="")
    annotate!((extent[1][1]+extent[1][2])/2, (extent[2][1]+extent[2][2])/2, text(label, color, :center, 10,))
end


function plot_cell_verify(extent, min_prob_value, max_prob_value, threshold; state_outlines_flag=false, use_color_wheel=true)
    if max_prob_value < min_prob_value
        # this may happen if synthesis is run with a convergence value > 1e-6
        max_prob_value = min_prob_value
    end
    x = [extent[1, 1], extent[1, 1], extent[1, 2], extent[1, 2]]
    y = [extent[2, 1], extent[2, 2], extent[2, 2], extent[2, 1]]
    shape = Plots.Shape(x, y)

    linealpha = state_outlines_flag ? 1.0 : 0.0

    color_wheel = cgrad([:red, :yellow2, :green], [0, 0.5, 0.95])
    if min_prob_value >= threshold
        plot!(shape, color=:white, fillalpha=1, linewidth=1, linecolor=:black, linealpha=linealpha, foreground_color_border=:white, foreground_color_axis=:white, label="")
        plot!(shape, color=get(color_wheel, 1), fillalpha=0.8, linewidth=1, linecolor=:black, linealpha=linealpha, foreground_color_border=:white, foreground_color_axis=:white, label="")
    elseif max_prob_value < threshold
        plot!(shape, color=get(color_wheel, 0), fillalpha=0.8, linewidth=1, linecolor=:black, linealpha=linealpha, foreground_color_border=:white, foreground_color_axis=:white, label="")
    else
        plot!(shape, color=:white, fillalpha=1, linewidth=1, linecolor=:black, linealpha=linealpha, foreground_color_border=:white, foreground_color_axis=:white, label="")
        if use_color_wheel
            plot!(shape, color=get(color_wheel, min_prob_value), fillalpha=0.8, linewidth=1, linecolor=:black, linealpha=linealpha, foreground_color_border=:white, foreground_color_axis=:white, label="")
        else
            plot!(shape, color=:yellow2, fillalpha=0.8, linewidth=1, linecolor=:black, linealpha=linealpha, foreground_color_border=:white, foreground_color_axis=:white, label="")
        end
    end
end


function plot_cell_shield(extent, num_actions, total_actions; state_outlines_flag=false)
    x = [extent[1, 1], extent[1, 1], extent[1, 2], extent[1, 2]]
    y = [extent[2, 1], extent[2, 2], extent[2, 2], extent[2, 1]]
    shape = Plots.Shape(x, y)

    linealpha = state_outlines_flag ? 1.0 : 0.0

    color_wheel = cgrad([:white, :green], [0, 1.0])

    plot!(shape, color=:white, fillalpha=1, linewidth=1, linecolor=:black, linealpha=linealpha, foreground_color_border=:white, foreground_color_axis=:white, label="")
    plot!(shape, color=get(color_wheel, num_actions/total_actions), fillalpha=0.8, linewidth=1, linecolor=:black, linealpha=linealpha, foreground_color_border=:white, foreground_color_axis=:white, label="")

end


function which_extent(extents, x, num_regions, num_dims)
    for idx in 1:(num_regions::Int)
        extent = extents[idx, :, :]
        in_extent = true
        for dim in 1:(num_dims::Int)
            if !(extent[dim, 1] < x[dim] <= extent[dim, 2])
                in_extent = false
                break
            end
        end
        if in_extent
            return idx
        end
    end
    return num_regions+1
end
