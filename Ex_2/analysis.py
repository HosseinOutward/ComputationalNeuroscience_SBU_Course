import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from Ex_2.Population import *


def generate_spike_data(pop, runtime, dt, conv_size = 10):
    spike_history = []
    idx_neuron = []
    neuron_type=[]
    for idx, neuron in enumerate(pop.neurons):
        idx_neuron += [idx for i in neuron.t_fired]

        type=('exc' if neuron.is_exc == 1 else 'inh')
        neuron_type += [type for i in neuron.t_fired]

        spike_history+=neuron.t_fired

    activity = np.bincount(np.array(np.array(spike_history)//dt, dtype = int))
    activity = np.pad(activity, (0, int(runtime//dt-len(activity))), 'constant')
    conv=int(conv_size * (.1 / dt))
    activity = np.convolve(activity, conv*[1/conv], "same")/len(pop.neurons)

    return spike_history, idx_neuron, neuron_type, activity


def plot_raster(pop, i_history, runtime, dt):
    spike_history, idx_neuron, neuron_type, activity = \
        generate_spike_data(pop, runtime, dt, conv_size=5)

    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(3, 1)
    raster = fig.add_subplot(gs[0, 0])
    raster.set_title("Raster plot")
    raster.set(ylabel="Neuron", xlabel="time(S)")
    sns.scatterplot(ax=raster, y=idx_neuron, x=spike_history, hue=neuron_type, marker='.')
    raster.set(xlim=(0, runtime))

    pop_activity = fig.add_subplot(gs[1, 0])
    pop_activity.set_title("Population activity")
    pop_activity.set(ylabel="A(t)", xlabel="time(S)")
    sns.lineplot(ax=pop_activity, x=np.arange(0, runtime, dt), y=activity)
    pop_activity.set(ylim=(0, 0.007))

    curr = fig.add_subplot(gs[2, 0])
    curr.set_title("Input current")
    curr.set(ylabel="I", xlabel="time(S)")
    sns.lineplot(ax=curr, x=np.arange(0, runtime, dt), y=i_history)

    fig.show()
