import matplotlib.pyplot as plt
import numpy as np
N_dims = [256, 512, 1024]
gflop = []
for N in N_dims:
    with open('run_sorken_'+str(N)+'_naive.txt') as f:
        print "Appending for ",N
        content = f.readlines()
        content = [line.strip() for line in content]
        content = [line.split('-')[2] for line in content]
        print "Content = ",len(content)
        content = list(map(float,content))
        content = [round(line, 5) for line in content]
        gflop.append(content)
print "GFlop = ",gflop
a = np.random.random((16, 16))
plt.imshow(a, cmap='hot', interpolation='nearest')
plt.show()
