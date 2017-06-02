import nengo
import numpy as np
from nengo import spa
import matplotlib.pyplot as plt

from spa_assoc_mem import AssociativeMemory
from transforms import associative_memory_transforms as am_transforms
from helpers import output_similarities_to_file as dump

seed = np.random.randint(100)

isi = 0.5
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
    ideal.number_one = spa.State(dimensions=dim)
    ideal.number_two = spa.State(dimensions=dim)
    ideal.inp = spa.Input(number_one=number_input, number_two=number_input2)
    
    ideal.assoc_mem = AssociativeMemory(input_vocab=vocab,
                                        output_vocab=vocab,
                                        input_keys=added_keys,
                                        output_keys=summed_keys,
                                        wta_output=True,
                                        wta_synapse=0.005,
                                        threshold=0.3)
    
    cortical_actions = spa.Actions(
        'assoc_mem = number_one * number_two'
    )
    
    ideal.cortical = spa.Cortical(cortical_actions)
    
    one_probe = nengo.Probe(ideal.number_one.output, synapse=0.03, label="one")
    two_probe = nengo.Probe(ideal.number_two.output, synapse=0.03, label="two")
    add_probe = nengo.Probe(ideal.assoc_mem.output, synapse=0.03, label="addition")


np.random.seed(seed)
with nengo.Simulator(ideal, seed=seed) as sim:
    sim.run(5.0)
dump(sim, vocab, "am_ideal")

am_decoders = am_transforms(ideal.assoc_mem.am, sim, 0.4, 0.05, 0.001)

with spa.SPA('damaged_addition', vocabs=[vocab], seed=seed) as model:
    model.number_one = spa.State(dimensions=dim)
    model.number_two = spa.State(dimensions=dim)
    model.inp = spa.Input(number_one=number_input, number_two=number_input2)
    
    model.assoc_mem = AssociativeMemory(input_vocab=vocab,
                                        output_vocab=vocab,
                                        input_keys=added_keys,
                                        output_keys=summed_keys,
                                        wta_output=True,
                                        wta_synapse=0.005,
                                        threshold=0.3,
                                        transforms=am_decoders)
    
    cortical_actions = spa.Actions(
        'assoc_mem = number_one * number_two'
    )
    
    model.cortical = spa.Cortical(cortical_actions)
    
    one_probe = nengo.Probe(model.number_one.output, synapse=0.03, label="one")
    two_probe = nengo.Probe(model.number_two.output, synapse=0.03, label="two")
    add_probe = nengo.Probe(model.assoc_mem.output, synapse=0.03, label="addition")

np.random.seed(seed)
with nengo.Simulator(model, seed=seed) as sim:
    sim.run(5.0)
dump(sim, vocab, "am_damaged")
