from Ex_1.Neurons import *
from numpy.random import normal as np_normal
from numpy.random import random as np_random
from numpy import arange
from numpy import array
import pickle


class FullyConnectedPopulation:
    def __init__(self, n_type, n_config, J, excit_count, inhib_count):
        self.J=J
        self.neurons = []

        for i in range(excit_count):
            self.neurons.append(eval('n_type(is_exc=True, ' + n_config + ')'))
        for i in range(inhib_count):
            self.neurons.append(eval('n_type(is_exc=False, ' + n_config + ')'))

        for pre_neuron in self.neurons:
            for post_neuron in self.neurons:
                if self.decide_to_connect() and post_neuron!=pre_neuron:
                    pre_neuron.post_syn.append((post_neuron, self.decide_weight()))

    def decide_to_connect(self):
        return True

    def decide_weight(self):
        return self.J / len(self.neurons) + np_normal(0.0, 0.001)

    def simulate_network_one_step(self, I_t):
        u_history=[]
        for neuron in self.neurons:
            inter_U, _ = neuron.simulate_one_step(I_t)
            u_history.append(inter_U)
        for neuron in self.neurons:
            neuron.syn_input = neuron.pre_syn_input
        if(neuron.internal_clock%20==0): print(neuron.internal_clock)
        return u_history


class FixedCouplingPopulation(FullyConnectedPopulation):
    def __init__(self, prob, *args, **kwargs):
        super(FullyConnectedPopulation, self).__init__(*args, **kwargs)
        self.prob = prob

    def decide_to_connect(self):
        return np_random() < self.prob

    def decide_weight(self):
        return self.J/len(self.neurons)/self.p + np_normal(0.0, 0.01)


class GaussianFullyConnected(FullyConnectedPopulation):
    def __init__(self, sigma, *args, **kwargs):
        super(FullyConnectedPopulation, self).__init__(*args, **kwargs)
        self.sigma = sigma

    def decide_weight(self):
        return np_normal(self.J/len(self.neurons), self.sigma/len(self.neurons))


class FullyConnectedPops(FullyConnectedPopulation):
    def __init__(self, J, pre_pop, post_pop):
        self.J=J
        self.pre_pop = pre_pop
        self.post_pop = post_pop
        self.neurons = pre_pop.neurons+post_pop.neurons

        for pre_neuron in self.pre_pop.neurons:
            for post_neuron in self.post_pop.neurons:
                if self.decide_to_connect() and post_neuron!=pre_neuron:
                    pre_neuron.post_syn.append((post_neuron, self.decide_weight()))

    def simulate_network_one_step(self, I_t):
        u_history=[]
        i_history=[]
        for neuron in self.neurons:
            inter_U, curr = neuron.simulate_one_step(I_t)
            u_history.append(inter_U)
            i_history.append(curr)
        for neuron in self.neurons:
            neuron.syn_input = neuron.pre_syn_input
        if neuron.internal_clock%20==0: print(neuron.internal_clock)
        return u_history, sum(i_history)/len(self.neurons)


if __name__ == "__main__":
    from Ex_1.analysis import random_smooth_array, plot_current, plot_mv_ms, limited_sin
    from Ex_2.analysis import *
    from math import sin

    dt=0.03125
    model = FullyConnectedPops(
        J=10,
        pre_pop=FullyConnectedPopulation(
            n_type=AELIF, excit_count=800, inhib_count=200, J=6.5,
            n_config="dt="+str(dt)+", R=10, tau=8, theta=-40, U_rest=-75, U_reset=-65, U_spike=5, "
                     "ref_period=2, ref_time=0, theta_rh=-45, delta_t=2, a=0.01, b=500, tau_k=100"),
        post_pop=FullyConnectedPopulation(
            n_type=LIF, excit_count=350, inhib_count=150, J=5.5,
            n_config="dt="+str(dt)+", R=10, tau=8, theta=-47, U_rest=-79, U_reset=-65, U_spike=5")
    )

    runtime=300; time_steps=int(runtime//dt)
    curr_func = lambda x: 1500*(sin(x/time_steps*4+2)+1.5)#limited_sin(time_steps)
    u_history=[]; i_history=[]
    plot_current([curr_func(t) for t in range(time_steps)], arange(0,runtime, dt))
    for t in range(time_steps):
        u, cur = model.simulate_network_one_step(curr_func(t*dt))
        u_history.append(u)
        i_history.append(cur)

    # *******************************
    import sys
    sys.setrecursionlimit(10000)

    with open("pop_data1.pickle", "wb") as f:
        pickle.dump(model.pre_pop, f)
    with open("pop_data2.pickle", "wb") as f:
        pickle.dump(model.post_pop, f)

    with open("pop_data1.pickle", "rb") as f:
        model.pre_pop = pickle.load(f)
    with open("pop_data2.pickle", "rb") as f:
        model.post_pop = pickle.load(f)

    plot_mv_ms(array(u_history)[:,1], arange(0,runtime, dt), top=-40, bottom=-80)
    plot_mv_ms(array(u_history)[:,-1], arange(0,runtime, dt), top=-40, bottom=-80)

    plot_raster(*generate_spike_data(model.pre_pop, runtime, dt), runtime, dt)
    plot_raster(*generate_spike_data(model.post_pop, runtime, dt), runtime, dt)



