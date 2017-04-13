import nengo
import numpy as np
from nengo import spa

from nengo.networks.assoc_mem import AssociativeMemory as AssocMem

isi = 0.6
dim = 128
number_keys = ['ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE', 'TEN', 'ELEVEN', 'TWELVE', 'THIRTEEN', 'FOURTEEN', 'FIFTEEN', 'SIXTEEN', 'SEVENTEEN', 'EIGHTEEN']

number_vocab = spa.Vocabulary(dim)
added_keys = []
summed_keys = []

for num in number_keys:
    number_vocab.parse(num)

for i in range(0, 9):
    for j in range(i, 9):
        ni = number_keys[i]
        nj = number_keys[j]
        number_vocab.add(ni+nj, number_vocab.parse("%s * %s" % (ni, nj)))
        added_keys.append(ni+nj)
        summed_keys.append(number_keys[i+j+1])

current_input = '0'
def number_input(t):
    global current_input
    
    isi_t = (t % isi)
    if isi_t < 0.002:
        current_input = number_keys[np.random.randint(0, 9)]
    elif (t % isi) <= 0.2:
        return current_input
    return '0'

with spa.SPA('AdditionMemory', seed=1) as model:
    # input
    model.number_in = spa.Buffer(dimensions=dim)
    model.inp = spa.Input(number_in=number_input)
    
    # addition associative memory
    model.assoc_mem = spa.AssociativeMemory(input_vocab=number_vocab,
                                            output_vocab=number_vocab,
                                            input_keys=added_keys,
                                            output_keys=summed_keys,
                                            wta_output=True,
                                            wta_synapse=0.005,
                                            threshold=0.3)
    
    # working memory
    model.mem = spa.Memory(dimensions=dim, subdimensions=16, 
                           synapse=0.1, neurons_per_dimension=50)

    # WTA memories to pull out 2 most recent items
    model.number_one = spa.AssociativeMemory(input_vocab=number_vocab,
                                             output_vocab=number_vocab,
                                             input_keys=number_keys,
                                             output_keys=number_keys,
                                             wta_output=True,
                                             wta_synapse=0.005,
                                             threshold=0.7)
                                             
    model.number_two = spa.AssociativeMemory(input_vocab=number_vocab,
                                             output_vocab=number_vocab,
                                             input_keys=number_keys,
                                             output_keys=number_keys,
                                             wta_output=True,
                                             wta_synapse=0.005,
                                             threshold=0.3)

    cortical_actions = spa.Actions(
        'mem = number_in',
        'number_one = mem',
        'number_two = mem - number_one',
        'assoc_mem = number_one * number_two',
    )
    
    model.cortical = spa.Cortical(cortical_actions)

