import nengo
import numpy as np
from nengo import spa

from nengo.networks.assoc_mem import AssociativeMemory as AssocMem

isi = 0.6
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

vocab.parse('POS1')
pos_next = vocab.add('POSN', vocab.create_pointer(unitary=True))

current_input = '0'
input_history = []
def number_input(t):
    global current_input
    
    isi_t = (t % isi)
    if isi_t < 0.002:
        current_input = single_digit[np.random.randint(0, 9)]
    else:
        input_history.append(current_input)
        return current_input
    input_history.append('0')
    return '0'
   
def control(t):
    init = (int(t // isi) == 0)
    if (t % isi) <= 0.2:
        return 'INPUT'
    elif (t % isi) <= 0.4:
        return 'ANSWER'
    return 'WAIT'

def position(t):
    result = 'POS1'
    for i in range(0, int(t // isi)):
        result += ' * POSN'
    return result
    
def answer(t):
    return single_digit[int(t // isi) % 9]

def answer_position(t):
    result = 'ANS'
    for i in range(0, int(t // isi)):
        result += ' * POSN'
    return result

with spa.SPA('AdditionMemory', seed=1) as model:
    # input
    model.number_in = spa.State(dimensions=dim)
    model.position = spa.State(dimensions=dim)
    model.control = spa.State(dimensions=dim)
    model.answer = spa.State(dimensions=dim)
    model.answer_position = spa.State(dimensions=dim)
    model.inp = spa.Input(number_in=number_input, position=position, control=control, answer=answer, answer_position=answer_position)
    
    # recency memory
    model.memory = spa.State(dimensions=dim, 
                              subdimensions=32, neurons_per_dimension=100,
                              feedback=1.0, feedback_synapse=0.1)

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
                                         input_keys=single_digit,
                                         output_keys=single_digit,
                                         wta_output=True,
                                         wta_synapse=0.005,
                                         threshold=0.3)

    model.prev_position = spa.State(dimensions=dim)
    model.memory_inp = spa.State(dimensions=dim)
    
    cortical_actions = spa.Actions(
        'prev_position = position * ~POSN',
        'memory = memory_inp',
        'one_am = memory * ~position',
        'two_am = memory * ~prev_position',
        'output = memory * ~answer_position',
    )
    model.cortical = spa.Cortical(cortical_actions)
    
    actions = spa.Actions(
        'dot(control, WAIT) --> memory_inp = 0',
        'dot(control, INPUT) --> memory_inp = number_in*position',
        'dot(control, ANSWER) --> memory_inp = answer*answer_position',
    )
    model.bg = spa.BasalGanglia(actions)
    model.thal = spa.Thalamus(model.bg)

    one_probe = nengo.Probe(model.one_am.output, synapse=0.03)
    two_probe = nengo.Probe(model.two_am.output, synapse=0.03)
    out_probe = nengo.Probe(model.output.output, synapse=0.03)

with nengo.Simulator(model) as sim:
    sim.run(2.4)
t = sim.trange()


# Output to file
one_data = sim.data[one_probe]
two_data = sim.data[two_probe]
out_data = sim.data[out_probe]

one_sim = spa.similarity(one_data, number_vocab)
two_sim = spa.similarity(two_data, number_vocab)
out_sim = spa.similarity(out_data, number_vocab)

with open('output.csv', 'w') as outfile:
    outfile.write("t,input,one,one_val,two,two_val,out,out_val\n")
    for i in range(0, len(one_sim)):
        one_max = np.argmax(one_sim[i])
        two_max = np.argmax(two_sim[i])
        out_max = np.argmax(out_sim[i])
        outfile.write("%f,%s,%s,%f,%s,%f,%s,%f\n" % (
                      i*0.001, input_history[i],
                      number_vocab.keys[one_max], one_sim[i][one_max],
                      number_vocab.keys[two_max], two_sim[i][two_max],
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

with open('out.csv', 'w') as outfile:
    header = 't,' + ','.join(number_vocab.keys) + '\n'
    outfile.write(header)
    for i in range(0, len(out_sim)):
        outfile.write(str(i * 0.001) + ',' + ','.join([str(s) for s in out_sim[i]]) + '\n')

