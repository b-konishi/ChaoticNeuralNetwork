import csv
import numpy as np
import math
import matplotlib.pyplot as plt
import warnings

class Analysis:
    def __init__(self, filename):
        self.time, self.x1, self.y1, self.x2, self.y2 = [],[],[],[],[]

        with open(filename, 'r') as f:
            reader = csv.reader(f)
            [next(reader) for _ in range(2)]
            index = next(reader)
            for d in reader:
                d = np.array(d).astype(np.float32)
                self.time.append(d[index.index('time[ms]')])
                self.x1.append(d[index.index('x1')])
                self.y1.append(d[index.index('y1')])
                self.x2.append(d[index.index('x2')])
                self.y2.append(d[index.index('y2')])
                # print(np.array(d).astype(np.float32))
        # print(self.time)
        print(self.lyapunov(self.time, [np.diff(self.x1), np.diff(self.y1)]))

    def lyapunov(self, time, data):
        plt.figure()
        
        # print((seqs))
        # print(self.normalize(seqs))

        interval = 200
        dt = 1/interval
        for seq in seqs:
            lyapunov = []
            for i in range(math.ceil(len(seq)/interval)):
                _seq = seq[i*interval:(i+1)*interval]
                diff = np.abs(np.diff(_seq))
                lyapunov.append(np.mean(np.log(diff/dt+1e-17)-np.log(2.0)))
            plt.plot(lyapunov)

        plt.show()

        return lyapunov

    def normalize(self, seqs):
        seqs = np.array(seqs).T

        # To avoid a warning about zero-divide
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            # Normalize [-0.5, 0.5]
            norm = (seqs-np.min(seqs,axis=0))/(np.max(seqs,axis=0)-np.min(seqs,axis=0))

        return np.where(np.isnan(norm), 0.0, norm)



if __name__ == '__main__':
    a = Analysis('../log.txt')




