import nengo
import numpy as np

from nengo import spa
from nengo.networks import CircularConvolution, InputGatedMemory

dim = 64
isi = 0.8
delivery_time = 0.4

number_keys = ['ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE', 'TEN', 'ELEVEN', 'TWELVE', 'THIRTEEN', 'FOURTEEN', 'FIFTEEN', 'SIXTEEN', 'SEVENTEEN', 'EIGHTEEN']
single_digit = number_keys[0:9]
added_keys = []
summed_keys = []

vocab = spa.Vocabulary(dim)
for num in number_keys:
    nv = vocab.parse(num)
    
for i in range(0, 9):
    for j in range(i, 9):
        ni = number_keys[i]
        nj = number_keys[j]
        vocab.add(ni+nj, vocab.parse("%s * %s" % (ni, nj)))
        added_keys.append(ni+nj)
        summed_keys.append(number_keys[i+j+1])

current_input = '0'
def number_list(t):
    global current_input

    if (t % isi) < 0.002:
        current_input = single_digit[np.random.randint(0, 9)]
    elif (t % isi) < delivery_time:
        return current_input
    return '0'

with spa.SPA('Indexing', vocabs=[vocab]) as model:
    model.number_input = spa.State(dimensions=dim)
    model.inp = spa.Input(number_input=number_list)

    model.input = spa.AssociativeMemory(input_vocab=vocab,
                                          output_vocab=vocab,
                                          input_keys=single_digit,
                                          output_keys=single_digit,
                                          wta_output=True)
    model.input_magnitude = spa.State(1)
    nengo.Connection(model.input.am.elem_output,
                     model.input_magnitude.input,
                     transform=np.ones((1, model.input.am.elem_output.size_out)),
                     synapse=0.005)
    nengo.Connection(model.number_input.output, model.input.input)
    
    model.input_mem = InputGatedMemory(200, dim)
    model.current_mem = InputGatedMemory(200, dim)
    model.previous_mem = InputGatedMemory(200, dim)
    
    model.clock_one = spa.State(1)
    model.one = nengo.Node(output=1)
    nengo.Connection(model.one, model.clock_one.input)
    nengo.Connection(model.input_magnitude.output, model.clock_one.input, transform=-1)
    
    model.clock_two = spa.State(1)
    nengo.Connection(model.input_magnitude.output, model.clock_two.input)
    
    nengo.Connection(model.clock_one.output, model.input_mem.gate)
    nengo.Connection(model.input.output, model.input_mem.input)
    
    nengo.Connection(model.clock_two.output, model.current_mem.gate)
    nengo.Connection(model.input_mem.output, model.current_mem.input)
    
    nengo.Connection(model.clock_one.output, model.previous_mem.gate)
    nengo.Connection(model.current_mem.output, model.previous_mem.input)
    
    model.current = spa.AssociativeMemory(input_vocab=vocab,
                                         output_vocab=vocab,
                                         input_keys=single_digit,
                                         output_keys=single_digit,
                                         wta_output=True)
    nengo.Connection(model.current_mem.output, model.current.input)
    
    model.previous = spa.AssociativeMemory(input_vocab=vocab,
                                           output_vocab=vocab,
                                           input_keys=single_digit,
                                           output_keys=single_digit,
                                           wta_output=True)
    nengo.Connection(model.previous_mem.output, model.previous.input)
    

    model.addition = spa.AssociativeMemory(input_vocab=vocab,
                                           output_vocab=vocab,
                                           input_keys=added_keys,
                                           output_keys=summed_keys,
                                           wta_output=True)

    model.cconv = nengo.networks.CircularConvolution(200, dimensions=dim)
    nengo.Connection(model.current.output, model.cconv.A)
    nengo.Connection(model.previous.output, model.cconv.B)
    nengo.Connection(model.cconv.output, model.addition.input)


