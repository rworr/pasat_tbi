import nengo
import numpy as np

def damage_decoders(decoders, distribution, pct_damage, severe_pct):
    s = decoders.shape
    fd = decoders.flatten(order='C')
    for i in range(0, len(fd)):
        r = np.random.random()
        if r <= severe_pct:
            fd[i] = 0
        elif np.random.random() <= distribution and fd[i] > 0.0:
            fd[i] = fd[i] + np.random.normal(0.0, pct_damage * abs(fd[i]))
    d = fd.reshape(s, order='C')
    return d

def working_memory_transforms(wm, sim, distribution=0.4, pct_damage=0.05, severe_pct=0):
    wm_conns = wm.connections
    ens_conns = [n.connections for n in wm.networks]

    wm_decoders = [sim.data[c].weights for c in wm_conns]
    ens_decoders = [[sim.data[c].weights for c in nc] for nc in ens_conns]

    for i in range(0, len(wm_decoders)):
        d = wm_decoders[i]
        if d.shape != ():
            wm_decoders[i] = damage_decoders(d, distribution, pct_damage, severe_pct)

    for e in range(0, len(ens_decoders)):
        for c in range(0, len(ens_decoders[e])):
            damaged = damage_decoders(ens_decoders[e][c], distribution, pct_damage, severe_pct)
            d_damaged = damage_decoders(damaged, distribution, pct_damage, severe_pct)
            ens_decoders[e][c] = d_damaged

    return wm_decoders, ens_decoders

def associative_memory_transforms(am, sim, distribution=0.4, pct_damage=0.05, severe_pct=0):
    am_conns = am.connections
    am_decoders = [sim.data[c].weights for c in am_conns]
    for i in range(0, len(am_decoders)):
        am_decoders[i] = damage_decoders(am_decoders[i], distribution, pct_damage, severe_pct)
    
    return am_decoders
