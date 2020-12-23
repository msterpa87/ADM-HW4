from random import randint

# The following implementation of HyperLogLog follows the algorithm
# outlined at page 4 https://hal.archives-ouvertes.fr/hal-00406166/

# constants
word_length = 128
bin_str_length = 32
buckets_bits = 6
primes = {32: int((2 ** 64 + 1) / 274177), 64: 2**127-1}


def rho(s):
	""" Returns the index of the first of from the left """
	try:
		return s.index('1') + 1
	except ValueError:
		return 1


def get_alpha(m):
	""" return the value alpha_m defined in the Flajolet et al.	"""
	assert(m in [16, 32, 64] or m >= 128)
	d = {16: 0.673, 32: 0.697, 64: 0.709}
	if m >= 128:
		return 0.7213/(1 + 1.079/m)
	else:
		return d[m]


def harmonic_mean(v):
	""" Computes the harmonic mean of vector v """
	return len(v) / sum([2 ** (-x) for x in v])


def get_hash_modular(bin_str_length=bin_str_length):
	""" Instantiate random parameters and returns a hash function

	Parameters
	----------
	bin_str_length : int
		length of strings return by the hash [32, 64]

	Returns
	-------
	function
		returns a hash function that takes int as inputs
	"""
	assert(bin_str_length in [32, 64])
	# seed(0)  # uncomment for reproducibility
	m = 2 ** bin_str_length
	p = primes[bin_str_length]
	max_val = 10**16

	# a,b integers mod p, a != 0
	a = randint(1, max_val) % p
	b = randint(0, max_val) % p

	def custom_hash(x):
		""" Hash function defined here https://en.wikipedia.org/wiki/Universal_hashing

		Parameters
		----------
		x : int

		Returns
		-------
		string
			returns a binary string representing the hash value
		"""
		bin_val = ((a * x + b) % p) % m
		bin_str = bin(bin_val)[2:]
		return "0" * (bin_str_length - len(bin_str)) + bin_str

	return custom_hash


class HyperLogLog(object):
	def __init__(self, bin_str_length=bin_str_length, buckets_bits=buckets_bits):
		self.buckets_bits = buckets_bits
		self.num_buckets = 2**buckets_bits
		self.buckets = [0] * self.num_buckets
		self.h = get_hash_modular(bin_str_length=bin_str_length)

		# first bits of hash will be used as bucket number
		# self.num_bins = bin_str_length - buckets_bits

	def add(self, s):
		""" Adds string s to the HLL structure """
		x = int(s, 16)
		x = self.h(x)
		i = int(x[:self.buckets_bits], 2) - 1  # bucket index
		pos = rho(x[self.buckets_bits:])  # 1-indexed position of first 1 from left

		# update bucket
		self.buckets[i] = max(pos, self.buckets[i])

	def __len__(self):
		""" Returns the number of elements in the structure """
		alpha = get_alpha(self.num_buckets)
		m = self.num_buckets
		avg = harmonic_mean(self.buckets)

		return int(alpha * m * avg)


def relative_accuracy(estimate, true_value=139000000):
	""" Returns the relative accuracy given the estimate """
	abs_error = abs(true_value - estimate)
	return abs_error / true_value


def estimate_distinct(filename="hash.txt"):
	"""
	Computes an estimate of the distinct elements of the
	multiset represented as line in filename

	:param filename: string
	:return: int
	"""
	hll = HyperLogLog()

	with open(filename, "r") as f:
		for line in f:
			line = line.strip()
			hll.add(line)

	return len(hll)
