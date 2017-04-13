import nengo
import numpy as np
from nengo import spa

from nengo.networks.assoc_mem import AssociativeMemory as AssocMem

dim = 64
numbers = ['ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE', 'TEN', 'ELEVEN', 'TWELVE', 'THIRTEEN', 'FOURTEEN', 'FIFTEEN', 'SIXTEEN', 'SEVENTEEN', 'EIGHTEEN']

number_vocab = spa.Vocabulary(dim)
added_vocab = spa.Vocabulary(dim)
added = []
summed = []

for num in numbers:
    number_vocab.parse(num)

for i in range(0, 9):
    for j in range(0, 9):
        ni = numbers[i]
        nj = numbers[j]
        added_vocab.parse(ni+nj)
        added.append(ni+nj)
        summed.append(numbers[i+j+1])


with spa.SPA('AdditionMemory', seed=1) as model:
    # create the AM module
    model.assoc_mem = spa.AssociativeMemory(input_vocab=added_vocab,
                                            output_vocab=number_vocab,
                                            input_keys=added,
                                            output_keys=summed)

    # present input to the AM
    model.am_input = spa.Input(assoc_mem='NINENINE')

