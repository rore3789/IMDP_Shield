# need to add CUDA, IntervalMDP, JuMP, Ipopt, Plots, PyCall, SpecialFunctions, ProgressBars, ColorSchemes, JLD, Distributions
# https://github.com/aria-systems-group/PosteriorBounds.jl

using Distributed

args = ARGS
addprocs(parse(Int, args[1]))

include("imdp_construction.jl")
include("bound_gp_outputs.jl")
include("visualize.jl")
include("shield_imdp.jl")
include("shield_refinement.jl")
using YAML

@pyimport pickle
py_object = pybuiltin("object")

exp_dir = @__DIR__
@info exp_dir

refinements = parse(Int, args[2])
refine_threshold = 1e-5

# TODO, make a lot of these user inputs
reuse_bounds = false  # feel free to edit any of these
reuse_policy = false
reuse_shield = false
reuse_refinement = false
test_shield = false
use_prob = false  # this determines if the error bounds used in IMDP construction are deterministic or probabilistic
threshold = 0.05
dyn_modes = nothing
fixed_dim = nothing
dfa_policy = nothing
time_horizon = 0  # how many time steps the shield provides guarantees for, 0 is infinite horizon. This is a hacky way to turn G(safe) into X(safe) & XX(safe)

# read from the yaml
yaml_file_name = args[3]
yaml_file_name = exp_dir * "/../" * yaml_file_name
yaml_data = YAML.load_file(yaml_file_name)

global_dir_name = yaml_data["global_dir_name"]
dyn_noise = yaml_data["process_dist"]["sig"]
noise_type = yaml_data["process_dist"]["dist"]
noise_info = [[-dyn_noise[dim], dyn_noise[dim], yaml_data["process_dist"]["var"][dim]] for dim in 1:length(dyn_noise)]
unsafe_set = yaml_data["unsafe_set"]
goal_set = yaml_data["goal_set"]
unsafe_label = yaml_data["unsafe_label"]
labels = yaml_data["labels"]
for k in keys(labels)
    labels[k] = yaml_data[labels[k]]
end

if global_dir_name == "example_2D_sys"
    # we can generate plots for this one since we have the dynamics already known
    dyn_modes = sys_2d_dynamics(dyn_noise[1])
end

dfa = pickle.load(open(exp_dir * yaml_data["dfa_path"], "r"))

global_exp_dir = exp_dir * "/../Systems/" * global_dir_name
general_data = numpy.load(global_exp_dir * "/general_info.npy")
nn_bounds_dir = global_exp_dir * "/nn_bounds"

plot_dir = global_exp_dir * "/plots"
if !isdir(plot_dir)
    mkdir(plot_dir)
end

num_dfa_states = length(dfa["states"])
if !isnothing(dfa["accept"])
    num_dfa_states -= 1
end
if !isnothing(dfa["sink"])
    num_dfa_states -= 1
end

num_modes = general_data[1]
num_dims = general_data[2]
num_sub_regions = general_data[3]
use_single_dim = general_data[5]
individual_nn = general_data[6]

for refinement in 0:refinements

    pimdp_filepath = global_exp_dir * "/pimdp_$(refinement).txt"
    res_filepath = global_exp_dir * "/shield_actions_$(refinement)"
    res_filepath_v = global_exp_dir * "/shield_values_$(refinement)"

    extents = numpy.load(global_exp_dir * "/extents_$refinement.npy")
    num_regions = size(extents)[1] - 1

    @info "The abstraction for refinement $refinement has $num_regions states"

    # define labels for every extent, this can be used to skip bounding obstacle posteriors
    label_fn = label_states(labels, extents, unsafe_label, num_dims, num_regions)

    # Get mean and sig bounds on gp
    @info "Bounding the GP mean and variance"
    reuse_check = true  # reuse_bounds && ((refinement == 0) || reuse_refinement)
    bound_gp_rkhs(num_regions, num_modes, num_dims, refinement, global_exp_dir, reuse_bounds, use_single_dim, dyn_noise, individual_nn)

    # setup imdp structure
    modes = [i for i in 1:num_modes]
    states = [i for i in 1:num_regions+1]
    imdp = IMDPModel(states, modes, nothing, nothing, label_fn, extents)

    policy = nothing  # There is code setup to use a NN policy to verify the shield but requires being able to propagate dynamics
    if (!isfile(res_filepath*".pkl") || !reuse_shield) || num_dims == 2
        @info "Constructing the PIMDP Model"
        ver_pimdp, syn_pimdp, p_action_diff, target_set, avoid_set, init_states = Simple_RKHS_Synthesis_PIMDP(extents, noise_info,
                                                        global_exp_dir, refinement, num_modes, num_regions, num_dims,
                                                        label_fn, dfa, imdp; noise_cdf=noise_type, use_prob=use_prob,
                                                        fixed_dim=fixed_dim)
    end

    imdp = nothing
    if !isfile(res_filepath*".pkl") || !reuse_shield
        @info "Generating the Shield"

        shield, values = shield_algorithm(syn_pimdp, avoid_set, threshold, eps)

        numpy.save(global_exp_dir*"/init_states_$refinement", init_states)
        numpy.save(global_exp_dir*"/p_act_diff_$refinement", p_action_diff)

        open(res_filepath*".pkl", "w") do f
            pickle.dump(shield, f, protocol=4)
        end
        open(res_filepath_v*".pkl", "w") do f
            pickle.dump(values, f, protocol=4)
        end
    else
        @info "Loading Shield"

        init_states = numpy.load(global_exp_dir*"/init_states_$refinement"*".npy")
        p_action_diff = numpy.load(global_exp_dir*"/p_act_diff_$refinement"*".npy")

        shield = open(res_filepath*".pkl", "r") do f
            pickle.load(f)
        end
        values = open(res_filepath_v*".pkl", "r") do f
            pickle.load(f)
        end
    end

    avg_modes = 0
    avg_reduced = 0
    num_reduced = 0
    num_zero = 0
    num_unsafe = 0
    for kk in 1:(num_regions::Int)
        label = label_fn[kk]
        if matches(label, "b")
            num_unsafe += 1
        end

        avail = length(shield[init_states[kk]])
        if avail < num_modes && !matches(label, "b")
            avg_reduced += avail
            num_reduced += 1
            if avail == 0 || values[init_states[kk]] > threshold
                num_zero += 1
            end
        end

        avg_modes += avail
    end
    @info "$(num_unsafe) states are labelled as unsafe in the IMDP and have no actions available"
    @info "An average of $(avg_modes/num_regions) actions are still available at each state"
    @info "$(num_regions - num_reduced - num_unsafe) states still have all actions available"
    @info "An average of $(avg_reduced/num_reduced) actions are still available at each state that has actions removed ($(num_reduced) of $(num_regions) states)"
    @info "$(num_zero) states are unsafe (no actions or high value), despite not being labelled unsafe initially"

    if num_dims == 2
        plot_shield_actions(shield, num_modes, extents, num_regions, num_dims, plot_dir, refinement, ver_pimdp,
                            time_horizon, values, threshold, init_states, policy;
                            labeled_regions=labels, obs_key=unsafe_label, num_dfa_states=num_dfa_states,
                            use_prob=use_prob, modes=dyn_modes, num_start_regions=100, simulate=test_shield,
                            dfa=dfa, dfa_policy=dfa_policy)
    end

# Below code not appropriately tested yet
#     if refinement < refinements
#         refinement_time = @elapsed begin
#             @info "Beginning refinement algorithm"
#             refine_filepath = global_exp_dir * "/refine_states_$(refinement)"
#             reuse_refine_states = false
#             if reuse_refine_states && isfile(refine_filepath * ".npy")
#                 refine_regions = numpy.load(refine_filepath * ".npy")
#             else
#                 refine_regions, dims_refined = find_specific_regions(extents, [0.1, nothing, nothing, nothing, nothing, nothing],
#                                                                      num_dims, num_regions; selected_dims=[1])
#                 numpy.save(refine_filepath, refine_regions)
#             end
#
#             refinement_algorithm(refine_regions, extents, modes, num_dims, global_dir_name, nn_bounds_dir, refinement;
#                                 threshold=refine_threshold, dims_refined=dims_refined, predefined_dims=true)
#
#
#
#         end
#         @info "Refined regions created in $(refinement_time) seconds"
#         print("\n")
#
#     else
#         @info "Done!"
#     end
end
