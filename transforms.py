import nengo
import numpy as np

def damage_decoders(decoders, distribution, pct_damage):
    s = decoders.shape
    fd = decoders.flatten(order='C')
    for i in range(0, len(fd)):
        if np.random.random() <= distribution and fd[i] > 0.0:
            fd[i] = fd[i] + np.random.normal(0.0, pct_damage * abs(fd[i]))
    d = fd.reshape(s, order='C')
    return d

def working_memory_transforms(wm, sim, distribution=0.4, pct_damage=0.05, loops=1):
    wm_conns = wm.connections
    ens_conns = [n.connections for n in wm.networks]

    wm_decoders = [sim.data[c].weights for c in wm_conns]
    ens_decoders = [[sim.data[c].weights for c in nc] for nc in ens_conns]

    for i in range(0, len(wm_decoders)):
        d = wm_decoders[i]
        if d.shape != ():
            wm_decoders[i] = damage_decoders(d, distribution, pct_damage)

    for e in range(0, len(ens_decoders)):
        for c in range(0, len(ens_decoders[e])):
            damaged = damage_decoders(ens_decoders[e][c], distribution, pct_damage)
            d_damaged = damage_decoders(damaged, distribution, pct_damage)
            ens_decoders[e][c] = d_damaged

    return wm_decoders, ens_decoders

def associative_memory_transforms(am, sim, distribution=0.4, pct_damage=0.05):
    am_conns = am.connections
    am_decoders = [sim.data[c].weights for c in am_conns]
    for i in range(0, len(am_decoders)):
        am_decoders[i] = damage_decoders(am_decoders[i], distribution, pct_damage)
    
    return am_decoders
