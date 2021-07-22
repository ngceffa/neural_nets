# import torch
# from torch import from_numpy as from_np
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import os


def fire_func(x):
    return torch.relu(torch.tanh(x))


def hinton(matrix):
    """Draw Hinton diagram for visualizing a weight matrix.
    (from original matplotlib docs)"""
    ax = plt.gca()
    max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor("black")
    ax.set_aspect("equal", "box")
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    # major_ticks_x = np.arange(-.5, matrix.shape[0], 1)
    # major_ticks_y = np.arange(-.5, matrix.shape[1], 1)
    # plt.xticks([])
    # plt.yticks([])
    # ax.set_xticks(major_ticks_x)
    # ax.set_yticks(major_ticks_y)
    plt.grid()

    for (x, y), w in np.ndenumerate(matrix):
        color = "white" if w > 0 else "black"
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle(
            [x - size / 2, y - size / 2], size, size, facecolor=color,
            edgecolor=color
        )
        ax.add_patch(rect)
    ax.autoscale_view()
    ax.invert_yaxis()
    plt.show()


def select_random(size, num):
    """Create a vector zeros.  Then entries are randomly selected to be 1.

    Args:
        size ([int]): 
        num ([int]): 

    Returns:
        [numpy array]:
    """
    array = np.zeros((size))
    random_neurons = np.random.choice(size, int(num), replace=False)
    array[random_neurons] = 1
    return array


def initrandW(W, scalefac=0.5):
    w_norm = W.astype(np.float32)
    for i in range(w_norm.shape[0]):
        normalizing_factor = np.sqrt(np.sum(w_norm[i, :] ** 2))
        if normalizing_factor > 0:
            w_norm[i, :] = w_norm[i, :] / normalizing_factor
    return scalefac * w_norm


def setPlottingColors(fontCol, backCol):
    # global plt
    plt.rcParams.update(
        {
            "lines.color": fontCol,
            "font.size": 16,
            "font.family": "Arial",
            "figure.figsize": [14, 10],
            "patch.edgecolor": fontCol,
            "text.color": fontCol,
            "axes.facecolor": backCol,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.edgecolor": "lightgray",
            "axes.labelcolor": fontCol,
            "axes.labelpad": 30.0,
            "xtick.color": fontCol,
            "ytick.color": fontCol,
            "grid.color": "lightgray",
            "figure.facecolor": backCol,
            "figure.edgecolor": backCol,
            "savefig.facecolor": backCol,
            "savefig.edgecolor": backCol,
        }
    )


def tens2np(tens):
    return np.asarray([ten.detach().numpy() for ten in tens])


def filtered_average_connectivity(
    path, cutoff_hemisphere_connections=2, cutoff_homologous_connections=9
    ):

    csv_connectivity = np.genfromtxt(
        f"{path}/connectivityMatrix.csv",
        dtype=np.uint8,
        delimiter=",",
        skip_header=1
    )
    connectivity_total = np.delete(csv_connectivity, 0, axis=1)
    # change pre->post to post->pre (very important to be consistent)
    connectivity_total = connectivity_total.T
    neuron_names = np.genfromtxt(
        f"{path}/connectivityMatrix.csv",
        dtype=np.unicode_,
        delimiter=",",
        skip_header=1,
        usecols=(0),
    ).tolist()
    neuron_names = [name.strip('"') for name in neuron_names]

    left_neurons_indices = [("LEFT" in name) for name in neuron_names]
    right_neurons_indices = [("RIGHT" in name) for name in neuron_names]
    left_neurons_names = [
        name.replace(" LEFT", "") for name in neuron_names if ("LEFT" in name)
    ]
    right_neurons_names = [
        name.replace(" RIGHT", "") for name in neuron_names if ("RIGHT" in name)
    ]
    assert left_neurons_names == right_neurons_names
    # Get connectivity values (between and within hemispheres)
    connectivity_left_left = connectivity_total[left_neurons_indices, :][
        :, left_neurons_indices
    ]
    connectivity_right_right = connectivity_total[right_neurons_indices, :][
        :, right_neurons_indices
    ]
    connectivity_left_right = connectivity_total[left_neurons_indices, :][
        :, right_neurons_indices
    ]
    connectivity_right_left = connectivity_total[right_neurons_indices, :][
        :, left_neurons_indices
    ]
    connectivity_left = connectivity_left_left + connectivity_left_right
    connectivity_right = connectivity_right_left + connectivity_right_right

    
    # Reduce the connectivity matrix:
    #   - hemi-hemi or hemi-other_hemisphere connections must be
    #        more than 2 for each neuron
    #   - homologous sum is more than 9 connections
    mask_same_hemisphere = (connectivity_left_left > cutoff_hemisphere_connections) * (
        connectivity_right_right > cutoff_hemisphere_connections
    )
    mask_opposite_hemispheres = (
        connectivity_left_right > cutoff_hemisphere_connections
    ) * (connectivity_right_left > cutoff_hemisphere_connections)
    hemispheres_mask = mask_same_hemisphere | mask_opposite_hemispheres
    homologous_mask = (
        connectivity_left + connectivity_right
    ) > cutoff_homologous_connections
    total_mask = hemispheres_mask * homologous_mask
    # here the starting weights are finally appearing
    connectivity = (total_mask) * (connectivity_left + connectivity_right) / 2
    return left_neurons_names, connectivity


def both_hemispheres_connectivity(path, cutoff=3):

    csv_connectivity = np.genfromtxt(
        f"{path}/connectivityMatrix.csv", dtype=np.uint8, delimiter=",", skip_header=1
    )
    connectivity_total = np.delete(csv_connectivity, 0, axis=1)
    # change pre->post to post->pre
    connectivity_total = connectivity_total.T
    neuron_names = np.genfromtxt(
        f"{path}/connectivityMatrix.csv",
        dtype=np.unicode_,
        delimiter=",",
        skip_header=1,
        usecols=(0),
    ).tolist()
    neuron_names = [name.strip('"') for name in neuron_names]

    left_neurons_indices = [("LEFT" in name) for name in neuron_names]
    right_neurons_indices = [("RIGHT" in name) for name in neuron_names]
    single_neurons_indices = [("SINGLE" in name) for name in neuron_names]
    left_neurons_names = [
        name.replace(" LEFT", "") for name in neuron_names if ("LEFT" in name)
    ]
    right_neurons_names = [
        name.replace(" RIGHT", "") for name in neuron_names if ("RIGHT" in name)
    ]
    single_neurons_names = [
        name.replace(" SINGLE", "") for name in neuron_names if ("SINGLE" in name)
    ]
    assert left_neurons_names == right_neurons_names
    # Get connectivity values (between and within hemispheres)
    # onlly reliable connections (usually if > 2):
    connectivity = connectivity_total[:, :] * (connectivity_total[:, :] > cutoff)
    # disregard autapses:
    print(
        "\nAutapses are ignored. Max autapses value was",
        np.amax(np.diag(connectivity)),
        ".",
    )
    np.fill_diagonal(connectivity, 0) # reject autapses

    return neuron_names, connectivity

def both_hemispheres_connectivity_allow_autapses(path, cutoff=3):

    csv_connectivity = np.genfromtxt(
        f"{path}/connectivityMatrix.csv", dtype=np.uint8, delimiter=",", skip_header=1
    )
    connectivity_total = np.delete(csv_connectivity, 0, axis=1)
    # change pre->post to post->pre
    connectivity_total = connectivity_total.T
    neuron_names = np.genfromtxt(
        f"{path}/connectivityMatrix.csv",
        dtype=np.unicode_,
        delimiter=",",
        skip_header=1,
        usecols=(0),
    ).tolist()
    neuron_names = [name.strip('"') for name in neuron_names]

    left_neurons_indices = [("LEFT" in name) for name in neuron_names]
    right_neurons_indices = [("RIGHT" in name) for name in neuron_names]
    single_neurons_indices = [("SINGLE" in name) for name in neuron_names]
    left_neurons_names = [
        name.replace(" LEFT", "") for name in neuron_names if ("LEFT" in name)
    ]
    right_neurons_names = [
        name.replace(" RIGHT", "") for name in neuron_names if ("RIGHT" in name)
    ]
    single_neurons_names = [
        name.replace(" SINGLE", "") for name in neuron_names if ("SINGLE" in name)
    ]
    assert left_neurons_names == right_neurons_names
    # Get connectivity values (between and within hemispheres)
    # onlly reliable connections (usually if > 2):
    connectivity = connectivity_total[:, :] * (connectivity_total[:, :] > cutoff)
    # disregard autapses:
    # print(
    #     "\nAutapses are ignored. Max autapses value was",
    #     np.amax(np.diag(connectivity)),
    #     ".",
    # )
    # np.fill_diagonal(connectivity, 0) # reject autapses

    return neuron_names, connectivity

def first_order_trial(
    valence,
    doUS,
    doC,
    input_size=70,
    input_fraction=0.1,
    total_time=80,
    dt=0.5,  # timestep
    cs_duraton=3,
    us_delay=2,
    us_duration=3,
    us_response=2,
    us_size=2,  #
    r_size=1,  #
    cs_start_window=[5, 15],
    cs_test_start_window=[65, 75],
    returntimes=False,
    ):
    # ORGANIZE TIMES SO THAT ALL DEPEND ON dt!!!!!!!!!
    # all given in rbitrary "time-units"
    # what is dus=int(2/dt) ? a duration of soeething... a delay?
    # valence must be 0 or 1 for positive or negative... ok---
    timesteps = int(total_time / dt)
    cs_duration = int(cs_duraton / dt)
    us_duration = int(us_duration / dt)
    us_response = int(us_response / dt)
    us_delay = int(us_delay / dt)
    # trial, time, neuron
    activated_neurons = np.zeros((timesteps, input_size))
    cs_start = int(np.random.randint(cs_start_window[0], cs_start_window[1]) / dt)
    cs_test_start = int(
        np.random.randint(cs_test_start_window[0], cs_test_start_window[1]) / dt
    )
    activated_subset = select_random(input_size, int(input_size * input_fraction))
    activated_subset_for_control = select_random(
        input_size, int(input_size * input_fraction)
    )
    activated_neurons[cs_start : (cs_start + cs_duration), :] = activated_subset
    unconditioned_stimulus = np.zeros((timesteps, us_size))
    # the us_size is basically always 2, since it codes for the valence... ?
    output = np.zeros((timesteps, r_size))
    # the r_size is basically always 1, since it codes for the output ?
    us_start = cs_start + us_delay
    if doC:
        activated_neurons[
            cs_test_start : (cs_test_start + cs_duration), :
        ] = activated_subset_for_control
    else:
        activated_neurons[
            cs_test_start : (cs_test_start + cs_duration), :
        ] = activated_subset
    if doUS:
        # assuming valence is 0 or 1 for positive/ngative
        unconditioned_stimulus[us_start : (us_start + us_duration), valence] = 1

    if doUS and not doC:
        output[cs_test_start : (cs_test_start + cs_duration), 0] = valence

    if returntimes:
        return (
            activated_neurons,
            unconditioned_stimulus,
            output,
            cs_start,
            us_start,
            cs_test_start,
        )
    return activated_neurons, unconditioned_stimulus, output


def first_order_trials(
    valence_values=[0, 1],
    doUS=True,
    doC=False,
    num_trials=2,
    input_size=70,
    input_fraction=0.1,
    total_time=80,
    dt=0.5,  # timestep
    cs_duraton=3,
    us_delay=2,
    us_duration=3,
    us_response=2,
    r_size=1,  #
    cs_start_window=[5, 15],
    cs_test_start_window=[65, 75],
    returntimes=False,
    ):
    assert [x > 0 for x in cs_start_window]
    assert [x < total_time for x in cs_start_window]
    assert [x > 0 for x in cs_test_start_window]
    assert [x < total_time for x in cs_test_start_window]
    # all given in rbitrary "time-units"
    # what is dus=int(2/dt) ? a duration of soeething... a delay?
    # valence must be 0 or 1 for positive or negative... ok---
    timesteps = int(total_time / dt)
    cs_duration = int(cs_duraton / dt)
    us_duration = int(us_duration / dt)
    us_response = int(us_response / dt)
    us_delay = int(us_delay / dt)
    us_size = len(valence_values)
    activated_neurons = np.zeros((num_trials, timesteps, input_size))
    cs_start = int(np.random.randint(cs_start_window[0], cs_start_window[1]) / dt)
    cs_test_start = int(
        np.random.randint(cs_test_start_window[0], cs_test_start_window[1]) / dt
    )
    activated_subset = select_random(input_size, int(input_size * input_fraction))
    activated_subset_for_control = select_random(
        input_size, int(input_size * input_fraction)
    )
    activated_neurons[:, cs_start : (cs_start + cs_duration), :] = activated_subset
    unconditioned_stimulus = np.zeros((num_trials, timesteps, us_size))
    # the us_size is basically always 2, since it codes for the valence... ?
    output = np.zeros((num_trials, timesteps, r_size))
    # the r_size is basically always 1, since it codes for the output ?
    us_start = cs_start + us_delay
    print(us_start, cs_start)
    for i in range(num_trials):
        valence = np.random.choice(valence_values)
        if doC:
            activated_neurons[
                i, cs_test_start : (cs_test_start + cs_duration), :
            ] = activated_subset_for_control
        else:
            activated_neurons[
                i, cs_test_start : (cs_test_start + cs_duration), :
            ] = activated_subset
        if doUS:
            if valence > 0:
                unconditioned_stimulus[i, us_start : (us_start + us_duration), 0] = 1
            else:
                unconditioned_stimulus[i, us_start : (us_start + us_duration), 1] = 1
        if doUS and not doC:
            output[i, cs_test_start : (cs_test_start + cs_duration), 0] = valence

    if returntimes:
        return (
            activated_neurons,
            unconditioned_stimulus,
            output,
            cs_start,
            us_start,
            cs_test_start,
        )
    return activated_neurons, unconditioned_stimulus, output


def first_order_trials_original_shape(
    valence_values=[-1, 1],
    doUS=True,
    doC=False,
    num_trials=10,
    input_size=70,
    input_fraction=0.1,
    total_time=80,
    dt=0.5,  # timestep
    cs_duraton=3,
    us_delay=2,
    us_duration=3,
    us_response=2,
    r_size=1,  #
    cs_start_window=[5, 15],
    cs_test_start_window=[65, 75],
    returntimes=False,
):
    assert [x > 0 for x in cs_start_window]
    assert [x < total_time for x in cs_start_window]
    assert [x > 0 for x in cs_test_start_window]
    assert [x < total_time for x in cs_test_start_window]
    # all given in rbitrary "time-units"
    # what is dus=int(2/dt) ? a duration of soeething... a delay?
    # valence must be 0 or 1 for positive or negative... ok---
    timesteps = int(total_time / dt)
    cs_duration = int(cs_duraton / dt)
    us_duration = int(us_duration / dt)
    us_response = int(us_response / dt)
    us_delay = int(us_delay / dt)
    us_size = len(valence_values)
    activated_neurons = np.zeros((timesteps, input_size, num_trials))
    cs_start = int(np.random.randint(cs_start_window[0], cs_start_window[1]) / dt)
    cs_test_start = int(
        np.random.randint(cs_test_start_window[0], cs_test_start_window[1]) / dt
    )

    # activated_neurons[cs_start:(cs_start + cs_duration), :, :] = \
    #                                                             activated_subset
    unconditioned_stimulus = np.zeros((timesteps, us_size, num_trials))
    # the us_size is basically always 2, since it codes for the valence... ?
    target_output = np.zeros((timesteps, num_trials))
    # the r_size is basically always 1, since it codes for the target_output ?
    us_start = cs_start + us_delay
    print(us_start, cs_start)
    for i in range(num_trials):
        valence = np.random.choice(valence_values)
        activated_subset = select_random(input_size, int(input_size * input_fraction))
        activated_subset_for_control = select_random(
            input_size, int(input_size * input_fraction)
        )
        print(valence)
        if doC:
            activated_neurons[
                cs_test_start : (cs_test_start + cs_duration), :, i
            ] = activated_subset_for_control
        else:
            activated_neurons[
                cs_test_start : (cs_test_start + cs_duration), :, i
            ] = activated_subset
        if doUS:
            if valence > 0:
                unconditioned_stimulus[us_start : (us_start + us_duration), 0, i] = 1
            else:
                unconditioned_stimulus[us_start : (us_start + us_duration), 1, i] = 1
        if doUS and not doC:
            target_output[cs_test_start : (cs_test_start + cs_duration), i] = valence

    print(np.amax(activated_neurons))
    if returntimes:
        return (
            activated_neurons,
            unconditioned_stimulus,
            target_output,
            cs_start,
            us_start,
            cs_test_start,
        )
    return (
        torch.from_numpy(activated_neurons),
        torch.from_numpy(unconditioned_stimulus),
        torch.from_numpy(target_output),
    )


def odor_valence_pairs(
    num_cells, num_odors=1, coding_level=0.1, valence_values=[-1, 1]
):
    odors = np.zeros((num_cells, num_odors))
    valences = np.zeros((num_odors))
    for i in range(num_odors):
        odors[:, i] = select_random(num_cells, int(num_cells * coding_level))
        valences[i] = np.random.choice(valence_values)
    return odors, valences


def first_order_xp_2(
    odors,  # vector of odors
    valences,  # vector of valences e.g. (-1, 1) for every odor
    valence_values=[-1, 1],
    doUS=True,
    doC=False,
    num_trials=30,
    input_size=70,
    input_fraction=0.1,
    total_time=80,
    dt=0.5,  # timestep
    cs_duraton=3,
    us_delay=2,
    us_duration=3,
    us_response=2,
    r_size=1,  #
    cs_start_window=[5, 15],
    cs_test_start_window=[65, 75],
    returntimes=False,
):
    assert [x > 0 for x in cs_start_window]
    assert [x < total_time for x in cs_start_window]
    assert [x > 0 for x in cs_test_start_window]
    assert [x < total_time for x in cs_test_start_window]
    # all given in rbitrary "time-units"
    # what is dus=int(2/dt) ? a duration of soeething... a delay?
    # valence must be 0 or 1 for positive or negative... ok---
    timesteps = int(total_time / dt)
    cs_duration = int(cs_duraton / dt)
    us_duration = int(us_duration / dt)
    us_response = int(us_response / dt)
    us_delay = int(us_delay / dt)
    us_size = len(valence_values)
    activated_KCs = np.zeros((timesteps, input_size, num_trials))
    cs_start = int(np.random.randint(cs_start_window[0], cs_start_window[1]) / dt)
    cs_test_start = int(
        np.random.randint(cs_test_start_window[0], cs_test_start_window[1]) / dt
    )

    unconditioned_stimulus = np.zeros((timesteps, us_size, num_trials))
    # the us_size is basically always 2, since it codes for the valence... ?
    target_output = np.zeros((timesteps, num_trials))
    # the r_size is basically always 1, since it codes for the target_output ?
    us_start = cs_start + us_delay

    for i in range(num_trials):  # num_trials = "batches"
        chosen_odor = np.random.choice(np.arange(0, odors.shape[1]))
        valence = valences[chosen_odor]
        activated_subset = odors[:, chosen_odor]
        activated_subset_for_control = odors[:, chosen_odor]
        activated_KCs[cs_start : (cs_start + cs_duration), :, i] = activated_subset
        if doC:
            activated_KCs[
                cs_test_start : (cs_test_start + cs_duration), :, i
            ] = activated_subset_for_control
        else:
            activated_KCs[
                cs_test_start : (cs_test_start + cs_duration), :, i
            ] = activated_subset
        if doUS:
            if valence > 0:
                unconditioned_stimulus[us_start : (us_start + us_duration), 0, i] = 1
            else:
                unconditioned_stimulus[us_start : (us_start + us_duration), 1, i] = 1
        if doUS and not doC:
            target_output[cs_test_start : (cs_test_start + cs_duration), i] = valence

    return (
        torch.from_numpy(activated_KCs),
        torch.from_numpy(unconditioned_stimulus),
        torch.from_numpy(target_output),
    )


def first_order_trials_xp(
    num_odors=10,
    valence_values=[-1, 1],
    doUS=True,
    doC=False,
    num_trials=10,  # batches
    input_size=70,
    input_fraction=0.1,
    total_time=80,
    dt=0.5,  # timestep
    cs_duraton=3,
    us_delay=2,
    us_duration=3,
    us_response=2,
    r_size=1,  #
    cs_start_window=[5, 15],
    cs_test_start_window=[65, 75],
):
    assert [x > 0 for x in cs_start_window]
    assert [x < total_time for x in cs_start_window]
    assert [x > 0 for x in cs_test_start_window]
    assert [x < total_time for x in cs_test_start_window]
    # all given in rbitrary "time-units"
    # what is dus=int(2/dt) ? a duration of soeething... a delay?
    # valence must be 0 or 1 for positive or negative... ok---
    timesteps = int(total_time / dt)
    cs_duration = int(cs_duraton / dt)
    us_duration = int(us_duration / dt)
    us_response = int(us_response / dt)
    us_delay = int(us_delay / dt)
    us_size = len(valence_values)
    activated_neurons = np.zeros((timesteps, input_size, num_trials))
    unconditioned_stimulus = np.zeros((timesteps, us_size, num_trials))
    cs_start = int(np.random.randint(cs_start_window[0], cs_start_window[1]) / dt)
    cs_test_start = int(
        np.random.randint(cs_test_start_window[0], cs_test_start_window[1]) / dt
    )
    # the us_size is basically always 2, since it codes for the valence... ?
    target_output = np.zeros((timesteps, num_trials))
    # the r_size is basically always 1, since it codes for the target_output ?
    us_start = cs_start + us_delay

    # odors
    odors = np.zeros((input_size, num_odors))
    valences = np.zeros((num_odors))
    for i in range(num_odors):
        odors[:, i] = select_random(input_size, int(input_size * input_fraction))
        valences[i] = np.random.choice(valence_values)

    for i in range(num_trials):
        odor_valence_pair = np.random.choice(np.arange(0, num_odors))
        valence = valences[odor_valence_pair]
        activated_subset = odors[:, odor_valence_pair]
        activated_subset_for_control = odors[:, odor_valence_pair]

        activated_neurons[cs_start : (cs_start + cs_duration), :, :] = activated_subset
        if doC:
            activated_neurons[
                cs_test_start : (cs_test_start + cs_duration), :, i
            ] = activated_subset_for_control
        else:
            activated_neurons[
                cs_test_start : (cs_test_start + cs_duration), :, i
            ] = activated_subset
        if doUS:
            if valence > 0:
                unconditioned_stimulus[us_start : (us_start + us_duration), 0, i] = 1
            else:
                unconditioned_stimulus[us_start : (us_start + us_duration), 1, i] = 1
        if doUS and not doC:
            target_output[cs_test_start : (cs_test_start + cs_duration), i] = valence

    return (
        torch.from_numpy(activated_neurons),
        torch.from_numpy(unconditioned_stimulus),
        torch.from_numpy(target_output),
    )


def first_order_single_xp(
    odor,  # subset of N{0, 1} encoding the odor
    valence,
    doUS=True,
    doC=False,
    input_size=70,  # num of KCs
    input_fraction=0.1,
    total_time=160,
    dt=0.5,  # timestep
    cs_duraton=3,
    us_delay=2,
    us_duration=3,
    us_response=2,
    cs_window=[5, 15],
    cs_test_window=[65, 75],
):

    assert [x > 0 for x in cs_window]
    assert [x < total_time for x in cs_window]
    assert [x > 0 for x in cs_test_window]
    assert [x < total_time for x in cs_test_window]

    timesteps = int(total_time / dt)
    cs_duration = int(cs_duraton / dt)
    us_duration = int(us_duration / dt)
    us_response = int(us_response / dt)
    us_delay = int(us_delay / dt)
    us_size = 2  # {-1, 1} only 2 possibilities
    KCs = np.zeros((timesteps, input_size))  # [T x num of KCs]
    # -> activation time traces
    cs_start = int(np.random.randint(cs_window[0], cs_window[1]) / dt)
    cs_test_start = int(np.random.randint(cs_test_window[0], cs_test_window[1]) / dt)
    unconditioned_stimulus = np.zeros((timesteps, us_size))
    # the us_size is basically always 2, since it codes for the valence... ?
    target_output = np.zeros((timesteps))
    # the r_size is basically always 1, since it codes for the target_output ?
    us_start = cs_start + us_delay
    odor_for_control = odor  # in this case it will probably not be used

    KCs[cs_start : (cs_start + cs_duration), :] = odor

    if doC:
        KCs[cs_test_start : (cs_start + cs_duration), :] = odor_for_control
    else:
        pass
        # KCs[cs_test_start:(cs_test_start + cs_duration), :] = odor

    if doUS:
        if valence > 0:
            unconditioned_stimulus[us_start : (us_start + us_duration), 0] = 1
        else:
            unconditioned_stimulus[us_start : (us_start + us_duration), 1] = 1

    if doUS and not doC:
        target_output[us_start : (us_start + us_duration)] = valence

    return (
        torch.from_numpy(KCs),
        torch.from_numpy(unconditioned_stimulus),
        torch.from_numpy(target_output),
    )


def extrapolated_connectivity(
    filepath, number_of_units, cutoff=2, average_method="mean"
):

    return None

def sparsity_index(matrix):
    M, N = matrix.shape[0], matrix.shape[1]
    nulls = 0
    for i in range(M):
        for j in range(N):
            if matrix[i, j] == 0:
                nulls +=1
    return nulls/(M*N)

def symmetry_index(matrix):
    symm = .5 * (matrix + np.transpose(matrix))
    asymm = .5 * (matrix - np.transpose(matrix))
    index = np.linalg.norm(asymm) / np.linalg.norm(symm)
    return index

if __name__ == "__main__":
    file_path = os.getcwd() + "/mb_modeling/eschbach2020/refactored_code/data"

    odor = np.zeros((70, 400))  # odor is encoded in a 70-long binary signal
    odor[0, :] = 1
    valence = np.ones((400))

    odors = np.zeros((70, 7))
    valences = np.zeros((7))
    for i in range(7):
        odors[:, i] = select_random(70, int(70 * 0.1))
        valences[i] = np.random.choice([-1, 1])

    plt.figure("od")
    plt.imshow(odors, aspect="auto")

    cs, us, tar = first_order_xp_2(odors, valences, num_trials=400)

    fig = plt.figure("1")
    fig.add_subplot(133)
    plt.title("target")
    plt.imshow(
        tar[:, :], origin="upper", interpolation="none", cmap="gray", aspect="auto"
    )
    plt.xlabel("trial")
    plt.ylabel("time")
    fig.add_subplot(131)
    plt.title("CS neuron 0")
    plt.imshow(
        cs[:, 0, :], origin="upper", interpolation="none", cmap="gray", aspect="auto"
    )
    plt.xlabel("trial")
    plt.ylabel("time")
    fig.add_subplot(132)
    plt.title("us (when positive)")
    plt.imshow(
        us[:, 0, :], origin="upper", interpolation="none", cmap="gray", aspect="auto"
    )
    plt.xlabel("trial")
    plt.ylabel("time")
    plt.show()
