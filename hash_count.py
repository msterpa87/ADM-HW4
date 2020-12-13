from hyperloglog2 import *
import hyperloglog
import sys

h = get_hash()
filename = "hash.txt"
verbose = True
print_freq = 10**7

my_hll = HyperLogLog()
hll = hyperloglog.HyperLogLog(0.01)

with open(filename, "r") as f:
    for i, line in enumerate(f):

        if verbose and i % print_freq == 0:
            print("[{}]".format(i))

        line = line.strip()

        hll.add(line)
        my_hll.add(line)

    my_estimate = len(my_hll)
    estimate = len(hll)
    diff = (my_estimate - estimate)/max(my_estimate, estimate)
    print("My estimate: ", my_estimate)
    print("Estimate: ", estimate)
    print("Diff: {:.0%}".format(diff))
