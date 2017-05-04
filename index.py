import nengo
import numpy as np

from nengo import spa

from nengo.networks import InputGatedMemory

dim = 512
isi = 1.0

vocab = spa.Vocabulary(dim)
vocab.parse('POS1')
vocab.add('NEXT', vocab.create_pointer(unitary=True))

for i in range(2, 61):
    vocab.add('POS%d' % i, vocab.parse('POS%d * NEXT' % (i-1)))

def inv_clock(t):
    if (t % isi) < 0.4:
        return 1
    return 0

def clock(t):
    if (t % isi) < 0.4:
        return 0
    return 1

def pos(t):
    if t < 1.0:
        return 'POS1'
    return '0'

with spa.SPA('Indexing', vocabs=[vocab]) as model:
    model.clock = nengo.Node(clock)
    model.inv_clock = nengo.Node(inv_clock)
    model.position = spa.State(dim)
    model.inp = spa.Input(position=pos)
    
    model.current = InputGatedMemory(800, dim)
    model.next = InputGatedMemory(800, dim)
    model.prev1 = spa.State(dim)
    model.prev2 = spa.State(dim)
    model.prev3 = spa.State(dim)
    model.prev4 = spa.State(dim)
    model.prev5 = spa.State(dim)
    
    model.current_input = spa.State(dim)
    model.next_input = spa.State(dim)
    model.current_output = spa.State(dim)
    model.next_output = spa.State(dim)


    nengo.Connection(model.clock, model.current.gate)
    nengo.Connection(model.inv_clock, model.next.gate)
    
    nengo.Connection(model.current_input.output, model.current.input)
    nengo.Connection(model.next_input.output, model.next.input)
    nengo.Connection(model.current.output, model.current_output.input)
    nengo.Connection(model.next.output, model.next_output.input)
    
    actions = spa.Actions(
        'dot(position, POS1) --> current_input = position',
        '0.7 --> current_input = next_output',
    )
    model.bg = spa.BasalGanglia(actions)
    model.thal = spa.Thalamus(model.bg)
    
    cortical_actions = spa.Actions(
        'next_input = current_output * NEXT',
        'prev1 = current_output * ~NEXT',
        'prev2 = prev1 * ~NEXT',
        'prev3 = prev2 * ~NEXT',
        'prev4 = prev3 * ~NEXT',
        'prev5 = prev4 * ~NEXT',
    )
    model.cortical = spa.Cortical(cortical_actions)

    current_probe = nengo.Probe(model.current_output.output, synapse=0.03, label="current")
    next_probe = nengo.Probe(model.next_output.output, synapse=0.03, label="next")
    prev1_probe = nengo.Probe(model.prev1.output, synapse=0.03, label="prev1")
    prev2_probe = nengo.Probe(model.prev2.output, synapse=0.03, label="prev2")
    prev3_probe = nengo.Probe(model.prev3.output, synapse=0.03, label="prev3")
    prev4_probe = nengo.Probe(model.prev4.output, synapse=0.03, label="prev4")
    prev5_probe = nengo.Probe(model.prev5.output, synapse=0.03, label="prev5")

    probes = [current_probe, next_probe]
    prev_probes = [prev1_probe, prev2_probe, prev3_probe, prev4_probe, prev5_probe]

with nengo.Simulator(model) as sim:
    sim.run(6.1)
t = sim.trange()

probe_data = [sim.data[probe] for probe in probes]
probe_sim = [spa.similarity(data, vocab) for data in probe_data]

with open('sim_data.csv', 'w') as outfile:
    outfile.write("t," + ','.join(["%s,%s_mag" % (p.label, p.label) for p in probes]) + '\n')
    for i in range(0, len(t)):
        max_idx = [np.argmax(psim[i]) for psim in probe_sim]
        line = ','.join(["%s,%f" % (vocab.keys[max_idx[idx]], probe_sim[idx][i][max_idx[idx]])
                         for idx in range(0, len(probes))])
        outfile.write("%f,%s\n" % (t[i], line))

for p in range(0, len(probes)):
    with open('%s.csv' % probes[p].label, 'w') as outfile:
        outfile.write('t,' + ','.join(vocab.keys) + '\n')
        for i in range(0, len(t)):
            outfile.write(str(t[i]) + ',' + ','.join([str(s) for s in probe_sim[p][i]]) + '\n')

with open('prev_sim.csv', 'w') as outfile:
    outfile.write("t,%s\n" % ','.join([p.label for p in prev_probes]))
    cur_data = sim.data[current_probe]
    for i in range(0, len(t)):
        prev_sim = [np.dot(cur_data[i], sim.data[p][i]) for p in prev_probes]
        outfile.write("%f,%s\n" % (t[i], ','.join([str(s) for s in prev_sim])))
