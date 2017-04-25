import nengo
import numpy as np
from nengo import spa
import matplotlib.pyplot as plt

from nengo.networks.assoc_mem import AssociativeMemory as AssocMem

init_phase = 0.3
isi = 0.8
dim = 512
number_keys = ['ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE', 'TEN', 'ELEVEN', 'TWELVE', 'THIRTEEN', 'FOURTEEN', 'FIFTEEN', 'SIXTEEN', 'SEVENTEEN', 'EIGHTEEN']
single_digit = number_keys[0:9]

number_vocab = spa.Vocabulary(dim)
vocab = spa.Vocabulary(dim)
added_keys = []
summed_keys = []

for num in number_keys:
    nv = vocab.parse(num)
    number_vocab.add(num, nv)

for i in range(0, 9):
    for j in range(i, 9):
        ni = number_keys[i]
        nj = number_keys[j]
        vocab.add(ni+nj, vocab.parse("%s * %s" % (ni, nj)))
        added_keys.append(ni+nj)
        summed_keys.append(number_keys[i+j+1])

vocab.parse('POS')
vocab.parse('ANS')
vocab.add('NEXT', vocab.create_pointer(unitary=True))

current_input = '0'
input_history = []
def number_input(t):
    global current_input
   
    ct = ((t - init_phase) % isi)
    if t < 0.002 or ct < 0.002:
        current_input = single_digit[np.random.randint(0, 9)]
    else:
        input_history.append(current_input)
        return current_input
    input_history.append('0')
    return '0'

def control(t):
    init = (t < init_phase)
    ct = ((t - init_phase) % isi)
    if ct <= 0.2 or init:
        return 'INPUT'
    return 'WAIT'

def position(t):
    result = 'POS'
    if t < init_phase:
        return result
    for i in range(0, int((t - init_phase) // isi)+1):
        result += ' * NEXT'
    return result
    
def answer(t):
    result = 'ANS'
    if t < init_phase:
        return result
    for i in range(0, int((t - init_phase) // isi)+1):
        result += ' * NEXT'
    return result

with spa.SPA('AdditionMemory', seed=1) as model:
    # input
    model.number_in = spa.State(dimensions=dim)
    model.position = spa.State(dimensions=dim)
    model.control = spa.State(dimensions=dim)
    model.answer = spa.State(dimensions=dim)
    model.inp = spa.Input(number_in=number_input, position=position, control=control, answer=answer)
    
    # recency memory
    model.memory = spa.State(dimensions=dim, 
                              subdimensions=64, neurons_per_dimension=100,
                              feedback=1.0, feedback_synapse=0.1)
  
    # addition associative memory
    model.assoc_mem = spa.AssociativeMemory(input_vocab=vocab,
                                            output_vocab=vocab,
                                            input_keys=added_keys,
                                            output_keys=summed_keys,
                                            wta_output=True,
                                            wta_synapse=0.005,
                                            threshold=0.3)
    
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
                                         threshold=0.2)

    model.output = spa.AssociativeMemory(input_vocab=vocab,
                                         output_vocab=vocab,
                                         input_keys=number_keys,
                                         output_keys=number_keys,
                                         wta_output=True,
                                         wta_synapse=0.005,
                                         threshold=0.3)
    
    model.prev_position = spa.State(dimensions=dim)
    model.memory_inp = spa.State(dimensions=dim)

    model.one_magnitude = spa.State(1)
    model.two_magnitude = spa.State(1)
    model.output_magnitude = spa.State(1)
    
    nengo.Connection(model.one_am.am.elem_output,
                     model.one_magnitude.input,
                     transform=np.ones((1, model.one_am.am.elem_output.size_out)),
                     synapse=0.005)
                     
    nengo.Connection(model.two_am.am.elem_output,
                     model.two_magnitude.input,
                     transform=np.ones((1, model.two_am.am.elem_output.size_out)),
                     synapse=0.005)
                     
    nengo.Connection(model.output.am.elem_output,
                     model.output_magnitude.input,
                     transform=np.ones((1, model.output.am.elem_output.size_out)),
                     synapse=0.005)


    cortical_actions = spa.Actions(
        'prev_position = position * ~NEXT',
        'memory = memory_inp',
        'one_am = memory * ~position',
        'two_am = memory * ~prev_position',
        'assoc_mem = one_am * two_am',
        'output = memory * ~answer',
    )
    model.cortical = spa.Cortical(cortical_actions)
    
    actions = spa.Actions(    
        'dot(control, INPUT) --> memory_inp = number_in * position', 
        'dot(control, WAIT) --> memory_inp = 0',
        #'dot(control, ANSWER) --> memory_inp = assoc_mem * answer',
        #'output_magnitude --> memory_inp = 0',
        #'one_magnitude + two_magnitude - 1 --> memory_inp = assoc_mem*answer',
        #'1.8 - one_magnitude --> memory_inp = number_in*position', 
        #'0.9 --> memory_inp = 0',
    )
    model.bg = spa.BasalGanglia(actions)
    model.thal = spa.Thalamus(model.bg)
    
    one_probe = nengo.Probe(model.one_am.output, synapse=0.03)
    two_probe = nengo.Probe(model.two_am.output, synapse=0.03)
    add_probe = nengo.Probe(model.assoc_mem.output, synapse=0.03)
    out_probe = nengo.Probe(model.output.output, synapse=0.03)

with nengo.Simulator(model) as sim:
    sim.run(24.3)
t = sim.trange()


# Output probes to file
one_data = sim.data[one_probe]
two_data = sim.data[two_probe]
add_data = sim.data[add_probe]
out_data = sim.data[out_probe]

one_sim = spa.similarity(one_data, number_vocab)
two_sim = spa.similarity(two_data, number_vocab)
add_sim = spa.similarity(add_data, number_vocab)
out_sim = spa.similarity(out_data, number_vocab)

with open('output.csv', 'w') as outfile:
    outfile.write("t,input,one,one_val,two,two_val,add,add_val,out,out_val\n")
    for i in range(0, len(one_sim)):
        one_max = np.argmax(one_sim[i])
        two_max = np.argmax(two_sim[i])
        add_max = np.argmax(add_sim[i])
        out_max = np.argmax(out_sim[i])
        outfile.write("%f,%s,%s,%f,%s,%f,%s,%f,%s,%f\n" % (
                      i*0.001, input_history[i],
                      number_vocab.keys[one_max], one_sim[i][one_max],
                      number_vocab.keys[two_max], two_sim[i][two_max],
                      number_vocab.keys[add_max], add_sim[i][add_max],
                      number_vocab.keys[out_max], out_sim[i][out_max],
                     ))

with open('one.csv', 'w') as outfile:
    header = 't,' + ','.join(number_vocab.keys) + '\n'
    outfile.write(header)
    for i in range(0, len(one_sim)):
        outfile.write(str(i * 0.001) + ',' + ','.join([str(s) for s in one_sim[i]]) + '\n')

with open('two.csv', 'w') as outfile:
    header = 't,' + ','.join(number_vocab.keys) + '\n'
    outfile.write(header)
    for i in range(0, len(two_sim)):
        outfile.write(str(i * 0.001) + ',' + ','.join([str(s) for s in two_sim[i]]) + '\n')

with open('add.csv', 'w') as outfile:
    header = 't,' + ','.join(number_vocab.keys) + '\n'
    outfile.write(header)
    for i in range(0, len(add_sim)):
        outfile.write(str(i * 0.001) + ',' + ','.join([str(s) for s in add_sim[i]]) + '\n')

with open('out.csv', 'w') as outfile:
    header = 't,' + ','.join(number_vocab.keys) + '\n'
    outfile.write(header)
    for i in range(0, len(out_sim)):
        outfile.write(str(i * 0.001) + ',' + ','.join([str(s) for s in out_sim[i]]) + '\n')
