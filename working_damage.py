import nengo
import numpy as np
from nengo import spa

from workingmem import WorkingMemory
from transforms import working_memory_transforms as wm_transforms
from helpers import output_similarities_to_file as dump

seed = 4

isi = 1.0
delivery_time = 0.4
dimensions = 64
number_keys = ['ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE', 'TEN', 'ELEVEN', 'TWELVE', 'THIRTEEN', 'FOURTEEN', 'FIFTEEN', 'SIXTEEN', 'SEVENTEEN', 'EIGHTEEN']
single_digit = number_keys[0:9]

vocab = spa.Vocabulary(dimensions)
for num in single_digit:
    nv = vocab.parse(num)

current_input = '0'
def number_input(t):
    global current_input

    if (t % isi) < 0.002:
        current_input = single_digit[np.random.randint(0, 9)]
    if (t % isi) < delivery_time:
        return current_input
    return '0'

def memory_clock(t):
    if (t % isi) < delivery_time:
        return 0
    return 1

with spa.SPA("ideal", vocabs=[vocab], seed=seed) as ideal:
    ideal.number = spa.State(dimensions)
    ideal.clock = nengo.Node(memory_clock)
    ideal.inp = spa.Input(number=number_input)
    
    ideal.mem = WorkingMemory(2000, dimensions)
    nengo.Connection(ideal.clock, ideal.mem.gate)
    ideal.out = spa.State(dimensions)
    
    actions = spa.Actions(
        "mem = number",
        "out = mem",
    )
    ideal.cortical = spa.Cortical(actions)

    input_p = nengo.Probe(ideal.number.output, synapse=0.03, label="input")
    output_p = nengo.Probe(ideal.out.output, synapse=0.03, label="output")

    conns = ideal.mem.connections
    e_conns = [n.connections for n in ideal.mem.networks]

np.random.seed(seed)
with nengo.Simulator(ideal, seed=seed) as sim:
    sim.run(10)

dump(sim, vocab, "ideal")

wm_decoders, ens_decoders = wm_transforms(ideal.mem, sim, 0.4, 0.05)

with spa.SPA("damaged", vocabs=[vocab], seed=seed) as model:
    model.number = spa.State(dimensions)
    model.clock = nengo.Node(memory_clock)
    model.inp = spa.Input(number=number_input)
    
    model.mem = WorkingMemory(2000, dimensions, transforms=wm_decoders, ens_transforms=ens_decoders)
    nengo.Connection(model.clock, model.mem.gate)
    model.out = spa.State(dimensions)
    
    actions = spa.Actions(
        "mem = number",
        "out = mem",
    )
    model.cortical = spa.Cortical(actions)

    input_p = nengo.Probe(model.number.output, synapse=0.03, label="input")
    output_p = nengo.Probe(model.out.output, synapse=0.03, label="output")

np.random.seed(seed)
with nengo.Simulator(model, seed=seed) as sim:
    sim.run(10)

dump(sim, vocab, "damaged")
