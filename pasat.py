import nengo
import numpy as np
from nengo import spa

from nengo.networks.assoc_mem import AssociativeMemory as AssocMem

isi = 1.0
dim = 128
number_keys = ['ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE', 'TEN', 'ELEVEN', 'TWELVE', 'THIRTEEN', 'FOURTEEN', 'FIFTEEN', 'SIXTEEN', 'SEVENTEEN', 'EIGHTEEN']
single_digit = number_keys[0:9]

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
vocab.parse('ANS')

current_input = '0'
def number_input(t):
    global current_input
    
    isi_t = (t % isi)
    if isi_t < 0.002:
        current_input = single_digit[np.random.randint(0, 9)]
    else:
        return current_input
    return '0'
    
def control(t):
    if int(t // isi) == 0:
        return 'INPUT'
    if (t % isi) <= 0.2:
        return 'INPUT'
    return '0'

def position(t):
    result = 'POS1'
    for i in range(0, int(t // isi)):
        result += ' * POSN'
    return result
    
with spa.SPA('AdditionMemory', seed=1) as model:
    # input
    model.number_in = spa.State(dimensions=dim)
    model.position = spa.State(dimensions=dim)
    model.control = spa.State(dimensions=dim)
    model.inp = spa.Input(number_in=number_input, position=position, control=control)
    
    # recency memory
    model.memory = spa.State(dimensions=dim, 
                              subdimensions=16, neurons_per_dimension=50,
                              feedback=1.0, feedback_synapse=0.1)
    # primacy memory
    #model.primacy_memory = spa.State(dimensions=dim, 
    #                          subdimensions=32, neurons_per_dimension=100,
    #                          feedback=-1.0, feedback_synapse=0.1)

    #model.memory = spa.State(dimensions=dim)
   
    # addition associative memory
    model.assoc_mem = spa.AssociativeMemory(input_vocab=vocab,
                                            output_vocab=vocab,
                                            input_keys=added_keys,
                                            output_keys=summed_keys,
                                            wta_output=True,
                                            wta_synapse=0.005,
                                            threshold=0.3)
    
    # WTA memories to pull out 2 most recent items
    model.one_am = spa.AssociativeMemory(input_vocab=vocab,
                                         output_vocab=vocab,
                                         input_keys=single_digit,
                                         output_keys=single_digit,
                                         wta_output=True,
                                         wta_synapse=0.005,
                                         threshold=0.3)
                                             
    model.two_am = spa.AssociativeMemory(input_vocab=vocab,
                                         output_vocab=vocab,
                                         input_keys=single_digit,
                                         output_keys=single_digit,
                                         wta_output=True,
                                         wta_synapse=0.005,
                                         threshold=0.2)

    model.output = spa.AssociativeMemory(input_vocab=vocab,
                                         output_vocab=vocab,
                                         input_keys=number_keys,
                                         output_keys=number_keys,
                                         wta_output=True,
                                         wta_synapse=0.005,
                                         threshold=0.3)

    model.prev_position = spa.State(dimensions=dim)
    model.memory_inp = spa.State(dimensions=dim)

    model.one_magnitude = spa.State(1)
    model.two_magnitude = spa.State(1)
    model.output_magnitude = spa.State(1)
    
    nengo.Connection(model.one_am.am.elem_output,
                     model.one_magnitude.input,
                     transform=np.ones((1, model.one_am.am.elem_output.size_out)),
                     synapse=0.005)
                     
    nengo.Connection(model.two_am.am.elem_output,
                     model.two_magnitude.input,
                     transform=np.ones((1, model.two_am.am.elem_output.size_out)),
                     synapse=0.005)
                     
    nengo.Connection(model.output.am.elem_output,
                     model.output_magnitude.input,
                     transform=np.ones((1, model.output.am.elem_output.size_out)),
                     synapse=0.005)

    cortical_actions = spa.Actions(
        'prev_position = position * ~POSN',
        #'recency_memory = memory_inp',
        #'primacy_memory = memory_inp',
        #'memory = recency_memory + primacy_memory',
        'memory = memory_inp',
        'one_am = memory * ~position',
        'output = memory',
        'two_am = memory * ~prev_position',
        'assoc_mem = one_am * two_am',
    )
    model.cortical = spa.Cortical(cortical_actions)
    
    actions = spa.Actions(
        'dot(control, INPUT) --> memory_inp = number_in*position',
        '1.5 - one_magnitude --> memory_inp = number_in*position',
        'one_magnitude + two_magnitude - 1 --> memory_inp = assoc_mem',
        'output_magnitude --> memory_inp = 0',
    )
    model.bg = spa.BasalGanglia(actions)
    model.thal = spa.Thalamus(model.bg)
    
        
