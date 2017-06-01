import numpy as np
from nengo import spa

def output_similarities_to_file(sim, vocab, prefix=""):
    probes = sim.model.probes
    t = sim.trange()

    probe_data = [sim.data[probe] for probe in probes]
    probe_sim = [spa.similarity(data, vocab) for data in probe_data]

    with open(prefix+'sim_data.csv', 'w') as outfile:
        outfile.write("t," + ','.join(["%s,%s_mag" % (p.label, p.label) for p in probes]) + '\n')
        for i in range(0, len(t)):
            max_idx = [np.argmax(psim[i]) for psim in probe_sim]
            line = ','.join(["%s,%f" % (vocab.keys[max_idx[idx]], probe_sim[idx][i][max_idx[idx]])
                             for idx in range(0, len(probes))])
            outfile.write("%f,%s\n" % (t[i], line))

    for p in range(0, len(probes)):
        with open(prefix+'%s.csv' % probes[p].label, 'w') as outfile:
            outfile.write('t,' + ','.join(vocab.keys) + '\n')
            for i in range(0, len(t)):
                outfile.write(str(t[i]) + ',' + ','.join([str(s) for s in probe_sim[p][i]]) + '\n')
