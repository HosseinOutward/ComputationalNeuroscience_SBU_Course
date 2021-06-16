from Ex_1.Neurons import *
from numpy.random import normal as np_normal
from numpy.random import random as np_random
from numpy import arange
from numpy import array
import pickle


class FullyConnectedPopulation:
    def __init__(self, n_type, n_config, J, excit_count, inhib_count, stdp_eng=None):
        self.J=J
        self.neurons = []
        self.conection_count=0
        self.stdp_eng=stdp_eng

        for i in range(excit_count):
            self.neurons.append(eval('n_type(is_exc=True, ' + n_config + ')'))
        for i in range(inhib_count):
            self.neurons.append(eval('n_type(is_exc=False, ' + n_config + ')'))

        self.create_network()

    def create_network(self):
        for pre_neuron in self.neurons:
            for post_neuron in self.neurons:
                self.connect_neurons(pre_neuron, post_neuron)

    def connect_neurons(self, pre_neuron, post_neuron):
        if self.decide_to_connect() and post_neuron != pre_neuron:
            self.conection_count += 1
            pre_neuron.post_syn.append([post_neuron, self.decide_weight()])
            post_neuron.pre_syn.append([len(pre_neuron.post_syn) - 1, pre_neuron])

    def decide_to_connect(self):
        return True

    def decide_weight(self):
        return self.J / self.conection_count + np_normal(0.0, 0.001)

    def draw_graph(self):
        import networkx as nx
        import pygraphviz
        import matplotlib.pyplot as plt
        ed=[]
        for pre_n_i, pre_neuron in enumerate(self.neurons):
            for post_neuron, w in pre_neuron.post_syn:
                ed.append([pre_n_i, self.neurons.index(post_neuron), w*1000//1/1000])
        G = nx.DiGraph()
        G.add_weighted_edges_from(ed)
        pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot', args="-Grankdir=LR")
        nx.draw(G,with_labels=True,pos=pos, font_weight='bold')

        edgewidth = [d['weight']*10 for (_, _, d) in G.edges(data=True)]
        # nx.draw_networkx_edges(G, pos, width=edgewidth)
        nx.draw_networkx_edges(G, pos, edge_color=edgewidth)

        plt.show()

    def fix_neurons(self, input_spike_list, output_spike_list):
        for neuron, to_spike in zip(
                self.input_neurons+self.output_neurons, input_spike_list+output_spike_list):
            neuron.syn_input=0;neuron.pre_syn_input=0
            if to_spike: neuron.U=neuron.U_spike+10
            elif neuron.U>neuron.U_reset: neuron.U=neuron.U_reset

    def simulate_network_one_step(self, I_t):
        u_history=[]
        i_history=[]
        for neuron in self.neurons:
            inter_U, curr = neuron.simulate_one_step(I_t)
            u_history.append(inter_U)
            i_history.append(curr)

        for neuron in self.neurons:
            neuron.syn_input += neuron.pre_syn_input
            neuron.pre_syn_input = 0
            if self.stdp_eng!=None and neuron.last_fired: self.stdp_eng.train(neuron)

        if neuron.internal_clock%20==0: print(neuron.internal_clock)

        return u_history, i_history


class FixedCouplingPopulation(FullyConnectedPopulation):
    def __init__(self, prob, *args, **kwargs):
        self.prob = prob
        super(FixedCouplingPopulation, self).__init__(*args, **kwargs)

    def decide_to_connect(self):
        return np_random() < self.prob

    def decide_weight(self):
        return self.J/self.conection_count/self.prob + np_normal(0.0, 0.01)


class GaussianFullyConnected(FullyConnectedPopulation):
    def __init__(self, sigma, *args, **kwargs):
        super(GaussianFullyConnected, self).__init__(*args, **kwargs)
        self.sigma = sigma

    def decide_weight(self):
        return np_normal(self.J/self.conection_count, self.sigma/self.conection_count)


# ************* 2 population *************
class FullyConnectedPops(FullyConnectedPopulation):
    def __init__(self, J, pre_pop, post_pop, decay=1.015, stdp_eng=None):
        self.J=J
        self.pre_pop = pre_pop
        self.post_pop = post_pop
        self.neurons = pre_pop.neurons+post_pop.neurons
        self.conection_count=0
        self.stdp_eng=stdp_eng
        self.decay=decay

        self.create_network()

    def create_network(self):
        for pre_neuron in self.pre_pop.neurons:
            for post_neuron in self.post_pop.neurons:
                self.connect_neurons(pre_neuron, post_neuron)


class FixedCouplingPops(FullyConnectedPops):
    def __init__(self, prob, *args, **kwargs):
        super(FixedCouplingPops, self).__init__(*args, **kwargs)
        self.prob = prob

    def decide_to_connect(self):
        return np_random() < self.prob

    def decide_weight(self):
        return self.J/self.conection_count/self.prob + np_normal(0.0, 0.01)


class GaussianFullyConnectedPops(FullyConnectedPops):
    def __init__(self, sigma, *args, **kwargs):
        self.sigma = sigma
        super(GaussianFullyConnectedPops, self).__init__(*args, **kwargs)

    def decide_weight(self):
        return np_normal(self.J/self.conection_count, self.sigma/self.conection_count)


if __name__ == "__main__":
    from Ex_1.analysis import *
    from Ex_2.analysis import *
    from math import sin

    dt=0.03125
    model = GaussianFullyConnectedPops(
        J=6, sigma=1,
        post_pop=FixedCouplingPopulation(
            n_type=AELIF, excit_count=50, inhib_count=5, J=6.5, prob=0.01,
            n_config="dt="+str(dt)+", R=10, tau=8, theta=-40, U_rest=-75, U_reset=-65, U_spike=5, "
                     "weight_sens=1, ref_period=2, ref_time=0, theta_rh=-45, delta_t=2, a=0.01, b=500, tau_k=100"),
        pre_pop=FullyConnectedPopulation(
            n_type=AELIF, excit_count=40, inhib_count=4, J=2.5,
            n_config="dt="+str(dt)+", R=10, tau=8, theta=-40, U_rest=-75, U_reset=-65, U_spike=5, "
                     "weight_sens=1, ref_period=2, ref_time=0, theta_rh=-45, delta_t=2, a=0.01, b=500, tau_k=100")
        )

    # model.draw_graph()
    runtime=300; time_steps=int(runtime//dt)
    curr_func = lambda x: 1515*(sin(x/time_steps*3.3+1)+1) # limited_sin(time_steps)
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

    plot_raster(*generate_spike_data(model.pre_pop, runtime, dt), runtime, dt, min_t=0)
    plot_raster(*generate_spike_data(model.post_pop, runtime, dt), runtime, dt, min_t=0)
