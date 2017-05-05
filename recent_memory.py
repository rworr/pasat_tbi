import nengo
import numpy as np
from nengo import spa

from nengo.networks import InputGatedMemory

dim = 32
isi = 0.6
delivery_time = 0.3
number_keys = ['ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE', 'TEN', 'ELEVEN', 'TWELVE', 'THIRTEEN', 'FOURTEEN', 'FIFTEEN', 'SIXTEEN', 'SEVENTEEN', 'EIGHTEEN']
single_digit = number_keys[0:9]

vocab = spa.Vocabulary(dim)
vocab.parse('CURRENT')
vocab.parse('PREVIOUS')
for num in number_keys:
    nv = vocab.parse(num)
    
current_input = '0'
def number_input(t):
    global current_input
    if (t % isi) < 0.002:
        current_input = single_digit[np.random.randint(0, 9)]
    elif (t % isi) < delivery_time:
        return current_input
    return '0'

with spa.SPA("Memory", vocabs=[vocab], seed=1) as model:
    model.number = spa.State(dim)
    model.inp = spa.Input(number=number_input)

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
    nengo.Connection(model.number.output, model.input.input)
    
    model.memory = spa.State(dimensions=dim)
    model.current = spa.AssociativeMemory(input_vocab=vocab,
                                          output_vocab=vocab,
                                          input_keys=single_digit,
                                          output_keys=single_digit,
                                          wta_output=True)
    model.previous = spa.AssociativeMemory(input_vocab=vocab,
                                           output_vocab=vocab,
                                           input_keys=single_digit,
                                           output_keys=single_digit,
                                           wta_output=True)
    
    model.cmem_one = InputGatedMemory(200, dim)
    model.cmem_two = InputGatedMemory(200, dim)
    
    model.one = nengo.Node(output=1)
    nengo.Connection(model.input_magnitude.output, model.cmem_one.gate, transform=-1)
    nengo.Connection(model.input_magnitude.output, model.cmem_two.gate)
    nengo.Connection(model.one, model.cmem_one.gate)
    
    model.current_out = spa.State(dim)
    
    nengo.Connection(model.current.output, model.cmem_one.input)
    nengo.Connection(model.cmem_one.output, model.cmem_two.input)
    nengo.Connection(model.cmem_two.output, model.current_out.input)
    
    actions = spa.Actions(
        'input_magnitude --> memory = input * CURRENT + current_out * PREVIOUS',
        '0.5 --> memory = current_out * CURRENT + previous * PREVIOUS',
    )
    model.bg = spa.BasalGanglia(actions)
    model.thal = spa.Thalamus(model.bg)
    
    cortical_actions = spa.Actions(
        'current = memory * ~CURRENT',
        'previous = memory * ~PREVIOUS',
    )
    model.cortical = spa.Cortical(cortical_actions)

    
    