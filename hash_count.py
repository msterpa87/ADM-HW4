from hyperloglog import *

h = get_hash()
filename = "hash.txt"
verbose = True
print_freq = 1000000

hll = HyperLogLog()

with open(filename, "r") as f:
    for i, line in enumerate(f):

        if verbose and i % print_freq == 0:
            print("[{}]".format(i))

        x = int(line.strip(), 16)
        x = h(x)
        hll.add(x)

    print("Estimate: ", hll.count())
