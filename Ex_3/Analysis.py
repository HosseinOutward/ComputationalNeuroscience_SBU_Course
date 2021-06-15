def plot_weight_spike(dt,runtime,pre_n, idx_post,w_his, max_x=0):
    from matplotlib import pyplot as plt
    plt.style.use('ggplot')
    a = pre_n.t_fired
    b = []
    c = []
    for i in range(len(a)):
        for j in range(500, 900):
            b.append(a[i])
            c.append(j / 1000)
    plt.scatter(b, c, marker='.', s=5, color='blueviolet')

    del a, b, c
    a = pre_n.post_syn[idx_post][0].t_fired
    b = []
    c = []
    for i in range(len(a)):
        for j in range(0, 400):
            b.append(a[i])
            c.append(j / 1000)
    plt.scatter(b, c, marker='.', s=5, color='deeppink')

    k = [a for i, a in enumerate(w_his) if i%int(1/dt)==0]

    plt.plot(k, color='maroon')
    plt.legend(['SynapticChange', 'PreNeuron', 'PostNeuron'], loc='upper left')
    plt.xlim(max_x, runtime)
    plt.show()
