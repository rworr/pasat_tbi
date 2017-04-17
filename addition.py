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

vocab.parse('POS1')
pos_next = vocab.add('POSN', vocab.create_pointer(unitary=True))

current_input = '0'
def number_input(t):
    global current_input
    
    isi_t = (t % isi)
    if isi_t < 0.002:
        current_input = number_keys[np.random.randint(0, 9)]
    elif isi_t <= 0.2:
        return current_input
    return '0'
    
current_input2 = '0'
def number_input2(t):
    global current_input2
    
    isi_t = (t % isi)
    if isi_t < 0.002:
        current_input2 = number_keys[np.random.randint(0, 9)]
    elif isi_t <= 0.2:
        return current_input2
    return '0'

with spa.SPA('AdditionMemory', seed=1) as model:
    # input
    model.number_in = spa.State(dimensions=dim)
    model.number_in2 = spa.State(dimensions=dim)
    model.inp = spa.Input(number_in=number_input, number_in2=number_input2)
    
    # addition associative memory
    model.assoc_mem = spa.AssociativeMemory(input_vocab=vocab,
                                            output_vocab=vocab,
                                            input_keys=added_keys,
                                            output_keys=summed_keys,
                                            wta_output=True,
                                            wta_synapse=0.005,
                                            threshold=0.3)
    
    # working memory
    model.seq_mem = spa.State(dimensions=dim, 
                              subdimensions=16, neurons_per_dimension=50,
                              feedback=1.0, feedback_synapse=0.1)

    # WTA memories to pull out 2 most recent items
    model.one_am = spa.AssociativeMemory(input_vocab=vocab,
                                         output_vocab=vocab,
                                         input_keys=number_keys,
                                         output_keys=number_keys,
                                         wta_output=True,
                                         wta_synapse=0.005,
                                         threshold=0.3)
                                             
    model.two_am = spa.AssociativeMemory(input_vocab=vocab,
                                         output_vocab=vocab,
                                         input_keys=number_keys,
                                         output_keys=number_keys,
                                         wta_output=True,
                                         wta_synapse=0.005,
                                         threshold=0.3)
    
    cortical_actions = spa.Actions(
        'one_am = number_in',
        'two_am = number_in2',
        'assoc_mem = one_am * two_am'
    )
    
    model.cortical = spa.Cortical(cortical_actions)
    
    #model.bg = spa.BasalGanglia(actions)
    #model.thal = spa.Thalamus(model.bg)
