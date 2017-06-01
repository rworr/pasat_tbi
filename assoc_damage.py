import nengo
import numpy as np
from nengo import spa
import matplotlib.pyplot as plt

from spa_assoc_mem import AssociativeMemory
from helpers import output_similarities_to_file as dump

seed = 4

isi = 0.4
dim = 64
number_keys = ['ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE', 'TEN', 'ELEVEN', 'TWELVE', 'THIRTEEN', 'FOURTEEN', 'FIFTEEN', 'SIXTEEN', 'SEVENTEEN', 'EIGHTEEN']

number_vocab = spa.Vocabulary(dim)
vocab = spa.Vocabulary(dim)
added_keys = []
summed_keys = []

for num in number_keys:
    v = vocab.parse(num)
    number_vocab.add(num, v)

for i in range(0, 9):
    for j in range(i, 9):
        ni = number_keys[i]
        nj = number_keys[j]
        vocab.add(ni+nj, vocab.parse("%s * %s" % (ni, nj)))
        added_keys.append(ni+nj)
        summed_keys.append(number_keys[i+j+1])

current_input = '0'
def number_input(t):
    global current_input
    
    isi_t = (t % isi)
    if isi_t < 0.002:
        current_input = number_keys[np.random.randint(0, 9)]
    elif isi_t <= 0.4:
        return current_input
    return '0'
    
current_input2 = '0'
def number_input2(t):
    global current_input2
    
    isi_t = (t % isi)
    if isi_t < 0.002:
        current_input2 = number_keys[np.random.randint(0, 9)]
    elif isi_t <= 0.4:
        return current_input2
    return '0'

with spa.SPA('ideal_addition', vocabs=[vocab], seed=seed) as ideal:
    # input
    ideal.number_one = spa.State(dimensions=dim)
    ideal.number_two = spa.State(dimensions=dim)
    ideal.inp = spa.Input(number_one=number_input, number_two=number_input2)
    
    # addition associative memory
    ideal.assoc_mem = AssociativeMemory(input_vocab=vocab,
                                        output_vocab=vocab,
                                        input_keys=added_keys,
                                        output_keys=summed_keys,
                                        wta_output=True,
                                        wta_synapse=0.005,
                                        threshold=0.3,
                                        label="addition_mem")
    
    cortical_actions = spa.Actions(
        'assoc_mem = number_one * number_two'
    )
    
    ideal.cortical = spa.Cortical(cortical_actions)
    
    one_probe = nengo.Probe(ideal.number_one.output, synapse=0.03, label="one")
    two_probe = nengo.Probe(ideal.number_two.output, synapse=0.03, label="two")
    add_probe = nengo.Probe(ideal.assoc_mem.output, synapse=0.03, label="three")

    am_conns = []
    for nt in ideal.assoc_mem.networks:
        if nt.label=="addition_mem":
            am_conns = nt.connections

with nengo.Simulator(ideal, seed=seed) as sim:
    sim.run(0.8)
dump(sim, vocab, "ideal")

decoders = [sim.data[c].weights for c in am_conns]
for i in range(0, len(decoders)):
    di = decoders[i]
    s = di.shape
    fd = di.flatten(order='C')
    for j in range(0, len(fd)):
        if np.random.random() <= 0.4:
            if fd[j] > 0:
                fd[j] = fd[j] + np.random.normal(0.0, 0.05*abs(fd[j]))
    decoders[i] = fd.reshape(s, order='C')

with spa.SPA('damaged_addition', vocabs=[vocab], seed=seed) as model:
    # input
    model.number_one = spa.State(dimensions=dim)
    model.number_two = spa.State(dimensions=dim)
    model.inp = spa.Input(number_one=number_input, number_two=number_input2)
    
    # addition associative memory
    model.assoc_mem = AssociativeMemory(input_vocab=vocab,
                                        output_vocab=vocab,
                                        input_keys=added_keys,
                                        output_keys=summed_keys,
                                        wta_output=True,
                                        wta_synapse=0.005,
                                        threshold=0.3,
                                        transforms=decoders,
                                        label="addition_mem")
    
    cortical_actions = spa.Actions(
        'assoc_mem = number_one * number_two'
    )
    
    model.cortical = spa.Cortical(cortical_actions)
    
    one_probe = nengo.Probe(model.number_one.output, synapse=0.03, label="one")
    two_probe = nengo.Probe(model.number_two.output, synapse=0.03, label="two")
    add_probe = nengo.Probe(model.assoc_mem.output, synapse=0.03, label="three")

with nengo.Simulator(model, seed=seed) as sim:
    sim.run(0.8)
dump(sim, vocab, "damaged")
