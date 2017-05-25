import nengo
import numpy as np
from nengo import spa

from nengo.networks import InputGatedMemory as WorkingMem
from nengo.networks.assoc_mem import AssociativeMemory as AssocMem

from helpers import output_similarities_to_file as dump

isi = 1.0
delivery_time = 0.4
answer_time = 0.7
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

vocab.add('CUR', vocab.create_pointer(unitary=True))
vocab.add('PREV', vocab.parse('CUR * CUR'))
vocab.parse('ANS')

current_input = '0'
def number_input(t):
    global current_input

    if (t % isi) < 0.002:
        current_input = single_digit[np.random.randint(0, 9)]
    if (t % isi) < delivery_time:
        return current_input
    return '0'

def updated_clock(t):
    if (t % isi) < delivery_time:
        return 0
    return 1
    
def memory_clock(t):
    if (t % isi) > delivery_time:
        return 0
    return 1

with spa.SPA("pasat", vocabs=[vocab]) as model:
    model.number = spa.State(dim)
    model.updated_clock = nengo.Node(updated_clock)
    model.memory_clock = nengo.Node(memory_clock)
    model.inp = spa.Input(number=number_input)

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
    
    model.recall = spa.State(dim)

    n = 3
    x = 1.0/np.sqrt(n)
    y = np.sqrt(1.0 - (1.0/n))

    cortical = spa.Actions(
        "updated_in = {0} * number * CUR + {1} * memory_out * CUR".format(x, y),
        "memory_in = {0} * addition * ANS + {1} * updated_out".format(x, y),
        "current = updated_out * ~CUR",
        "previous = updated_out * ~PREV",
        "addition = current * previous",
        "recall = memory_out * ~ANS",
    )

    model.cortical = spa.Cortical(cortical)
    nengo.Connection(model.updated_clock, model.updated.gate)
    nengo.Connection(model.memory_clock, model.memory.gate)

    input_probe = nengo.Probe(model.number.output, synapse=0.03, label="input")
    cur_probe = nengo.Probe(model.current.output, synapse=0.03, label="current")
    prev_probe = nengo.Probe(model.previous.output, synapse=0.03, label="previous")
    add_probe = nengo.Probe(model.addition.output, synapse=0.03, label="addition")
    recall_probe = nengo.Probe(model.recall.output, synapse=0.03, label="recall")

with nengo.Simulator(model) as sim:
    sim.run(6)
t = sim.trange()

dump(sim, vocab)
