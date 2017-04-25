import nengo
import numpy as np
from nengo import spa
import matplotlib.pyplot as plt

from nengo.networks.assoc_mem import AssociativeMemory as AssocMem

isi = 0.4
dim = 512
number_keys = ['ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE', 'TEN', 'ELEVEN', 'TWELVE', 'THIRTEEN', 'FOURTEEN', 'FIFTEEN', 'SIXTEEN', 'SEVENTEEN', 'EIGHTEEN']

number_vocab = spa.Vocabulary(dim)
vocab = spa.Vocabulary(dim)
added_keys = []
summed_keys = []

for num in number_keys:
    v = vocab.parse(num)
    number_vocab.add(num, v)

for i in range(0, 9):
    for j in range(i, 9):
        ni = number_keys[i]
        nj = number_keys[j]
        vocab.add(ni+nj, vocab.parse("%s * %s" % (ni, nj)))
        added_keys.append(ni+nj)
        summed_keys.append(number_keys[i+j+1])

current_input = '0'
def number_input(t):
    global current_input
    
    isi_t = (t % isi)
    if isi_t < 0.002:
        current_input = number_keys[np.random.randint(0, 9)]
    elif isi_t <= 0.2:
        return current_input
    return '0'
    
current_input2 = '0'
def number_input2(t):
    global current_input2
    
    isi_t = (t % isi)
    if isi_t < 0.002:
        current_input2 = number_keys[np.random.randint(0, 9)]
    elif isi_t <= 0.2:
        return current_input2
    return '0'

with spa.SPA('AdditionMemory', seed=1) as model:
    # input
    model.number_in = spa.State(dimensions=dim)
    model.number_in2 = spa.State(dimensions=dim)
    model.inp = spa.Input(number_in=number_input, number_in2=number_input2)
    
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
                                         input_keys=number_keys,
                                         output_keys=number_keys,
                                         wta_output=True,
                                         wta_synapse=0.005,
                                         threshold=0.3)
                                             
    model.two_am = spa.AssociativeMemory(input_vocab=vocab,
                                         output_vocab=vocab,
                                         input_keys=number_keys,
                                         output_keys=number_keys,
                                         wta_output=True,
                                         wta_synapse=0.005,
                                         threshold=0.3)
    
    cortical_actions = spa.Actions(
        'one_am = number_in',
        'two_am = number_in2',
        'assoc_mem = one_am * two_am'
    )
    
    model.cortical = spa.Cortical(cortical_actions)
    
    one_probe = nengo.Probe(model.one_am.output, synapse=0.03)
    two_probe = nengo.Probe(model.two_am.output, synapse=0.03)
    add_probe = nengo.Probe(model.assoc_mem.output, synapse=0.03)

with nengo.Simulator(model) as sim:
    sim.run(0.8)
t = sim.trange()


# Output all data to file
one_data = sim.data[one_probe]
two_data = sim.data[two_probe]
add_data = sim.data[add_probe]

one_sim = spa.similarity(one_data, number_vocab)
two_sim = spa.similarity(two_data, number_vocab)
add_sim = spa.similarity(add_data, number_vocab)

with open('output.csv', 'w') as outfile:
    outfile.write("t,one,one_val,two,two_val,add,add_val\n")
    for i in range(0, len(one_sim)):
        one_max = np.argmax(one_sim[i])
        two_max = np.argmax(two_sim[i])
        add_max = np.argmax(add_sim[i])
        outfile.write("%f,%s,%f,%s,%f,%s,%f,\n" % (
                      i*0.001,
                      number_vocab.keys[one_max], one_sim[i][one_max],
                      number_vocab.keys[two_max], two_sim[i][two_max],
                      number_vocab.keys[add_max], add_sim[i][add_max],
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
