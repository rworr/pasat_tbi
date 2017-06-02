import numpy as np

import nengo
from ensemblearray import EnsembleArray
from nengo.solvers import LstsqL2
from nengo.spa.module import Module

from nengo import spa

class WorkingMemory(Module):
    """Stores a given vector in memory, with input controlled by a gate.

    Parameters
    ----------
    n_neurons : int
        Number of neurons per dimension in the vector.
    dimensions : int
        Dimensionality of the vector.

    feedback : float, optional (Default: 1.0)
        Strength of the recurrent connection from the memory to itself.
    difference_gain : float, optional (Default: 1.0)
        Strength of the connection from the difference ensembles to the
        memory ensembles.
    recurrent_synapse : float, optional (Default: 0.1)

    difference_synapse : Synapse (Default: None)
        If None, ...
    kwargs
        Keyword arguments passed through to ``nengo.Network``.

    Returns
    -------
    net : Network
        The newly built memory network, or the provided ``net``.

    Attributes
    ----------
    net.diff : EnsembleArray
        Represents the difference between the desired vector and
        the current vector represented by ``mem``.
    net.gate : Node
        With input of 0, the network is not gated, and ``mem`` will be updated
        to minimize ``diff``. With input greater than 0, the network will be
        increasingly gated such that ``mem`` will retain its current value,
        and ``diff`` will be inhibited.
    net.input : Node
        The desired vector.
    net.mem : EnsembleArray
        Integrative population that stores the vector.
    net.output : Node
        The vector currently represented by ``mem``.
    net.reset : Node
        With positive input, the ``mem`` population will be inhibited,
        effectively wiping out the vector currently being remembered.

    """

    def __init__(self, n_neurons, dimensions,
                 feedback=1.0, difference_gain=1.0, synapse=0.1,
                 transforms=None, ens_transforms=None, 
                 label=None, seed=None, add_to_container=None):
        super(WorkingMemory, self).__init__(label, seed, add_to_container)
        
        vocab = dimensions
        n_total_neurons = n_neurons * dimensions

        if transforms is None:
            transforms = iter([feedback, -1.0, difference_gain, np.ones((n_total_neurons, 1)) * -10])
        else:
            transforms = iter(transforms)

        if ens_transforms is None:
            ens_transforms = iter([None, None])
        else:
            ens_transforms = iter(ens_transforms)

        with self:
            # integrator to store value
            self.mem = EnsembleArray(n_neurons, dimensions, label="mem", transforms=ens_transforms.next())
            nengo.Connection(self.mem.output, self.mem.input,
                             transform=transforms.next(),
                             synapse=synapse)

            # calculate difference between stored value and input
            self.diff = EnsembleArray(n_neurons, dimensions, label="diff", transforms=ens_transforms.next())
            nengo.Connection(self.mem.output, self.diff.input, transform=transforms.next())

            # feed difference into integrator
            nengo.Connection(self.diff.output, self.mem.input,
                             transform=transforms.next(),
                             synapse=synapse)

            # gate difference (if gate==0, update stored value,
            # otherwise retain stored value)
            self.gate = nengo.Node(size_in=1)
            self.diff.add_neuron_input()
            nengo.Connection(self.gate, self.diff.neuron_input,
                             transform=transforms.next(),
                             synapse=None)

            self.input = self.diff.input
            self.output = self.mem.output

            self.inputs = dict(default=(self.input, vocab))
            self.outputs = dict(default=(self.output, vocab))
