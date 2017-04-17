import nengo
import numpy as np
from nengo import spa

from nengo.networks.assoc_mem import AssociativeMemory as AssocMem

isi = 0.6
dim = 128
number_keys = ['ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE', 'TEN', 'ELEVEN', 'TWELVE', 'THIRTEEN', 'FOURTEEN', 'FIFTEEN', 'SIXTEEN', 'SEVENTEEN', 'EIGHTEEN']

vocab = spa.Vocabulary(dim)
added_keys = []
summed_keys = []

for num in number_keys:
    vocab.parse(num)

for i in range(0, 9):
    for j in range(i, 9):
        ni = number_keys[i]
        nj = number_keys[j]
        vocab.add(ni+nj, vocab.parse("%s * %s" % (ni, nj)))
        added_keys.append(ni+nj)
        summed_keys.append(number_keys[i+j+1])

input_one = '0'
def number_one(t):
    global input_one
    
    isi_t = (t % isi)
    if isi_t < 0.002:
        input_one = number_keys[np.random.randint(0, 9)]
    elif isi_t <= 0.2:
        return input_one
    return '0'

input_two = '0'
def number_two(t):
    global input_two
    
    isi_t = (t % isi)
    if isi_t < 0.002:
        input_two = number_keys[np.random.randint(0, 9)]
    elif isi_t <= 0.2:
        return input_tw
    return '0'

with spa.SPA('AdditionMemory', seed=1) as model:
    # input
    model.number_one = spa.State(dimensions=dim)
    model.number_two = spa.State(dimensions=dim)
    model.inp = spa.Input(number_one=input_one, number_two=input_two)
    
    # addition associative memory
    model.assoc_mem = spa.AssociativeMemory(input_vocab=vocab,
                                            output_vocab=vocab,
                                            input_keys=added_keys,
                                            output_keys=summed_keys,
                                            wta_output=True,
                                            wta_synapse=0.005,
                                            threshold=0.3)
    
    cortical_actions = spa.Actions(
        'assoc_mem = number_one * number_two'
    )
    
    model.cortical = spa.Cortical(cortical_actions)
