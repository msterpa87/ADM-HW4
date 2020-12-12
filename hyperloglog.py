from math import log
from random import seed, randint
import re

# constants
word_length = 128
bin_str_length = 32
buckets_bits = 4


def get_alpha(m):
	"""

	:param m: int
	:return: the constant to compute estimate in HLL structure
	"""
	assert(m in [16, 32, 64] or m >= 128)
	d = {16: 0.673, 32: 0.697, 64: 0.709}
	if m >= 128:
		return 0.7213/(1 + 1.079/m)
	else:
		return d[m]


def harmonic_mean(v):
	"""
	Computes the harmonic mean of vector v

	:param v: vector of ints
	:return: int
	"""
	return len(v)/sum([1/(2**x) for x in v])


def trail_zeroes(string):
	"""

	:param string:
	:return:
	"""
	m = re.match(r'^(0+)', string)
	if m is None:
		return 0
	else:
		return len(m[0])


def get_hash(bin_str_length=bin_str_length, word_length=word_length):
	"""

	:param bin_str_length:
	:param word_length:
	:return:
	"""
	num_bins = 2 ** bin_str_length
	max_a = 2 ** word_length
	max_b = 2**(word_length - bin_str_length)
	max_hash_length = word_length - bin_str_length
	a = randint(1, max_a-1)
	b = randint(1, max_b-1)

	def custom_hash(x):
		"""

		:param x: int
		:return: binary string
		"""
		bin_str = bin(((a * x + b) % max_a) >> (word_length - bin_str_length))[2:]
		return "0" * (bin_str_length - len(bin_str)) + bin_str

	return custom_hash


class HyperLogLog(object):
	def __init__(self, bin_str_length=bin_str_length, buckets_bits=buckets_bits):
		self.buckets_bits = buckets_bits
		self.num_buckets = 2**buckets_bits
		self.buckets = [0] * self.num_buckets

		# first bits of hash will be used as bucket number
		# self.num_bins = bin_str_length - buckets_bits

	def add(self, x):
		"""
		adds x to the HLL structure

		:param x: binary string
		:return: None
		"""
		i = int(x[:self.buckets_bits], 2) - 1  # bucket number
		zeroes = trail_zeroes(x[self.buckets_bits:])  # trailing zeroes

		# update bucket
		self.buckets[i] = max(zeroes, self.buckets[i])

	def count(self):
		alpha = get_alpha(self.num_buckets)
		m = self.num_buckets
		avg = harmonic_mean(self.buckets)

		return int(alpha * m * avg)

