import numpy as np
import sys

l = []

for x in sys.argv[1:]:
    l.append(np.genfromtxt(x).item())
    
a = np.array(l)
print('\pgfmathsetmacro{{\eqavg}}{{{:.2f}}}'.format(np.mean(a)))
print('\pgfmathsetmacro{{\eqstd}}{{{:.2f}}}'.format(np.std(a)))
