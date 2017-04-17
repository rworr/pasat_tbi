import nengo
import numpy as np
from nengo import spa

from nengo.networks.assoc_mem import AssociativeMemory as AssocMem

isi = 0.6
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
        #vocab.add(ni+nj, vocab.parse("%s * %s" % (ni, nj)))
        added_keys.append(ni+nj)
        summed_keys.append(number_keys[i+j+1])

vocab.parse('POS1')
pos_next = vocab.add('POSN', vocab.create_pointer(unitary=True))

current_input = '0'
def number_input(t):
    global current_input
    
    isi_t = (t % isi)
    if isi_t < 0.002:
        current_input = single_digit[np.random.randint(0, 9)]
    elif (int(t // isi) == 0):
        return current_input
    elif isi_t <= 0.2:
        return current_input
    return '0'
    
def control(t):
    init = (int(t // isi) == 0)
    if (t % isi) <= 0.2:
        return 'INPUT'
    elif (t % isi) <= 0.4:
        return 'WAIT'
    elif (t % isi) <= 0.5:
        if init:
            return 'INPUT'
        return 'ANSWER'
    else:
        return 'WAIT'

def position(t):
    result = 'POS1'
    for i in range(0, int(t // isi)):
        result += ' * POSN'
    return result
    
def answer(t):
    return single_digit[int(t // isi) % 9]

with spa.SPA('AdditionMemory', seed=1) as model:
    # input
    model.number_in = spa.State(dimensions=dim)
    model.position = spa.State(dimensions=dim)
    model.control = spa.State(dimensions=dim)
    model.answer = spa.State(dimensions=dim)
    model.inp = spa.Input(number_in=number_input, position=position, control=control, answer=answer)
    
    # recency memory
    model.recency_memory = spa.State(dimensions=dim, 
                              subdimensions=32, neurons_per_dimension=100,
                              feedback=1.0, feedback_synapse=0.1)
    # primacy memory
    model.primacy_memory = spa.State(dimensions=dim, 
                              subdimensions=32, neurons_per_dimension=100,
                              feedback=-1.0, feedback_synapse=0.1)

    model.memory = spa.State(dimensions=dim)

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
                                         threshold=0.3)

    model.output = spa.AssociativeMemory(input_vocab=vocab,
                                         output_vocab=vocab,
                                         input_keys=single_digit,
                                         output_keys=single_digit,
                                         wta_output=True,
                                         wta_synapse=0.005,
                                         threshold=0.3)

    model.prev_position = spa.State(dimensions=dim)
    model.memory_inp = spa.State(dimensions=dim)
    
    model.one_enc = spa.State(dimensions=dim)
    model.no_one = spa.State(dimensions=dim)
    
    cortical_actions = spa.Actions(
        'prev_position = position * ~POSN',
        'recency_memory = memory_inp',
        'primacy_memory = memory_inp',
        'memory = recency_memory + primacy_memory',
        'one_am = memory * ~position',
        'one_enc = one_am * position',
        'no_one = memory - one_enc',
        'output = no_one * ~ANS',
        'two_am = memory * ~prev_position',
    )
    model.cortical = spa.Cortical(cortical_actions)
    
    actions = spa.Actions(
        'dot(control, WAIT) --> memory_inp = 0',
        'dot(control, INPUT) --> memory_inp = number_in*position',
        'dot(control, RECALL) --> memory_inp = one_am*position',
        'dot(control, ANSWER) --> memory_inp = answer*ANS',
    )
    model.bg = spa.BasalGanglia(actions)
    model.thal = spa.Thalamus(model.bg)
