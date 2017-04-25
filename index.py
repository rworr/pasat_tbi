import nengo
import numpy as np

from nengo import spa

dim = 32

vocab = spa.Vocabulary(dim)
vocab.parse('POSITION')
vocab.add('NEXT', vocab.create_pointer(unitary=True))

with spa.SPA('Indexing') as model:
    osc = nengo.Ensemble(100, 2)
    nengo.Connection(osc, osc, transform=[[1, -0.1], [0.1, 1]])