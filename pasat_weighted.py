import nengo
import numpy as np
from nengo import spa

from nengo.networks import InputGatedMemory as WorkingMem
from nengo.networks.assoc_mem import AssociativeMemory as AssocMem

from helpers import output_similarities_to_file as dump

isi = 1.0
delivery_time = 0.4
dim = 64
number_keys = ['ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE', 'TEN', 'ELEVEN', 'TWELVE', 'THIRTEEN', 'FOURTEEN', 'FIFTEEN', 'SIXTEEN', 'SEVENTEEN', 'EIGHTEEN']
single_digit = number_keys[0:9]

vocab = spa.Vocabulary(dim)
added_keys = []
summed_keys = []

for num in number_keys:
    nv = vocab.parse(num)

for i in range(0, 9):
    for j in range(i, 9):
        ni = number_keys[i]
        nj = number_keys[j]
        vocab.add(ni+nj, vocab.parse("%s * %s" % (ni, nj)))
        added_keys.append(ni+nj)
        summed_keys.append(number_keys[i+j+1])

vocab.add('POS', vocab.create_pointer(unitary=True))
vocab.parse('ANS')

current_input = '0'
def number_input(t):
    global current_input

    if (t % isi) < 0.002:
        current_input = single_digit[np.random.randint(0, 9)]
    if (t % isi) < delivery_time:
        return current_input
    return '0'

def control(t):
    if (t < isi) or (t % isi) < delivery_time:
        return 'INPUT'
    return 'WAIT'

def position(t):
    result = 'POS'
    for i in range(0, int(t // isi)):
        result += ' * POS'
    return result

def input_mag(t):
    if (t % isi) < delivery_time:
        return 1
    return 0
    
def ninput_mag(t):
    if (t % isi) < delivery_time:
        return 0
    return 1

with spa.SPA("pasat", vocabs=[vocab], seed=1) as model:
    model.position = spa.State(dim)
    model.control = spa.State(dim)
    model.number = spa.State(dim)
    model.clock1 = nengo.Node(input_mag)
    model.clock2 = nengo.Node(ninput_mag)
    model.inp = spa.Input(position=position, control=control, number=number_input)

    # addition memory
    model.addition = spa.AssociativeMemory(input_vocab=vocab,
                                            output_vocab=vocab,
                                            input_keys=added_keys,
                                            output_keys=summed_keys,
                                            wta_output=True,
                                            wta_synapse=0.005,
                                            threshold=0.3)

    # WTA memories to pull out 2 most recent items
    model.current = spa.AssociativeMemory(input_vocab=vocab,
                                         output_vocab=vocab,
                                         input_keys=single_digit,
                                         output_keys=single_digit,
                                         wta_output=True,
                                         wta_synapse=0.005,
                                         threshold=0.3)

    model.previous = spa.AssociativeMemory(input_vocab=vocab,
                                         output_vocab=vocab,
                                         input_keys=single_digit,
                                         output_keys=single_digit,
                                         wta_output=True,
                                         wta_synapse=0.005,
                                         threshold=0.2)
    
    model.memory = WorkingMem(2000, dim)
    model.memory_in = spa.State(dim)
    model.memory_out = spa.State(dim)
    nengo.Connection(model.memory_in.output, model.memory.input)
    nengo.Connection(model.memory.output, model.memory_out.input)
    
    model.updated = WorkingMem(2000, dim)
    model.updated_in = spa.State(dim)
    model.updated_out = spa.State(dim)
    nengo.Connection(model.updated_in.output, model.updated.input)
    nengo.Connection(model.updated.output, model.updated_out.input)

    model.prev_position = spa.State(dim)
    
    x = 1/np.sqrt(2)
    cortical = spa.Actions(
        "updated_in = {0} * number * position + {0} * memory_out".format(x),
        "memory_in = updated_out",
        "current = memory_out * ~position",
        "prev_position = position * ~POS",
        "previous = memory_out * ~prev_position",
        "addition = current * previous",
    )
    model.cortical = spa.Cortical(cortical)
    nengo.Connection(model.clock2, model.updated.gate)
    nengo.Connection(model.clock1, model.memory.gate)

    input_probe = nengo.Probe(model.number.output, synapse=0.03, label="input")
    cur_probe = nengo.Probe(model.current.output, synapse=0.03, label="current")
    prev_probe = nengo.Probe(model.previous.output, synapse=0.03, label="previous")
    add_probe = nengo.Probe(model.addition.output, synapse=0.03, label="addition")

with nengo.Simulator(model) as sim:
    sim.run(30)
t = sim.trange()

dump(sim, vocab)
