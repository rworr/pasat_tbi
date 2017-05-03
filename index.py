import nengo
import numpy as np

from nengo import spa
from nengo.networks import InputGatedMemory

dim = 32
isi = 0.6

vocab = spa.Vocabulary(dim)
vocab.parse('POS1')
vocab.add('NEXT', vocab.create_pointer(unitary=True))
vocab.add('POS2', vocab.parse('POS1 * NEXT'))

def clock(t):
    if (t % isi) < 0.3:
        return 0
    return 1

def pos(t):
    if t < 0.6:
        return 'POS1'
    if t < 1.2:
        return 'POS2'
    return '0'

with spa.SPA('Indexing') as model:
    model.clock = nengo.Node(clock)
    model.position = spa.State(dim)
    model.output = spa.State(dim)
    model.inp = spa.Input(position=pos)
    
    
    model.current = InputGatedMemory(200, dim)
    
    nengo.Connection(model.clock, model.current.gate)
    nengo.Connection(model.position.output, model.current.input)
    nengo.Connection(model.current.output, model.output.input)
    