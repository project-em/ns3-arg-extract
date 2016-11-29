import numpy as np

def average_precision(correct_set, guess_list, k):
	if (k < len(guess_list)):
		guess_list = guess_list[:k]
	count_correct = 0
	sum_precision  = 0
	relevant_song_count = len(correct_set)
	for i in range(1, len(guess_list) + 1):
		if guess_list[i - 1] in correct_set:
			correct_set.remove(guess_list[i - 1])
			count_correct += 1
			precision = np.true_divide(count_correct, i)
			sum_precision += precision
	# change in recall
	if len(guess_list) < relevant_song_count:
		return np.true_divide(sum_precision, len(guess_list))
	else:
		return np.true_divide(sum_precision, relevant_song_count)

if __name__ == '__main__':
	print "should be 0.25: ", average_precision([1,2,3,4,5],[6,4,7,1,2], 2)
	print "should be 0.2: ", average_precision([1,2,3,4,5],[1,1,1,1,1], 5)
	print "should be 1: ", average_precision([1,2,3,4,5], [1,2,3,4,5], 5)