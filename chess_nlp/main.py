from utils.utils import get_database_as_sentences
from chess import Move, Board
import pickle

if __name__ == "__main__":

	pgn_filepath = "chess_data/sample.pgn"
	# pgn_filepath = "chess_data/lichess_db_standard_rated_2013-01.pgn"

	unigram_filepath = "bigram/unigram_counts.pkl"
	bigram_filepath = "bigram/bigram_counts.pkl"
	trigram_filepath = "bigram/trigram_counts.pkl"

	sents, vocab = get_database_as_sentences(pgn_filepath)

	unigram_counts = {}
	bigram_counts = {}
	trigram_counts = {}

	for sent in sents:
		corpus = sent.split(' ')
		size = len(corpus)

		for i in range(size):
			unigram = corpus[i]
			if(unigram in unigram_counts):
				unigram_counts[unigram] += 1
			else:
				unigram_counts[unigram] = 1
		
		for i in range(size - 1):
			bigram = (corpus[i], corpus[i+1])
			if(bigram in bigram_counts):
				bigram_counts[bigram] += 1
			else:
				bigram_counts[bigram] = 1

		for i in range(size - 2):
			trigram = (corpus[i], corpus[i+1], corpus[i+2])
			if(trigram in trigram_counts):
				trigram_counts[trigram] += 1
			else:
				trigram_counts[trigram] = 1

	print(f"Saving data to {unigram_filepath}, {bigram_filepath}, {trigram_filepath}")

	with open(unigram_filepath, "wb") as wf:
		pickle.dump(unigram_counts, wf)

	with open(bigram_filepath, "wb") as wf:
		pickle.dump(bigram_counts, wf)

	with open(trigram_filepath, "wb") as wf:
		pickle.dump(trigram_counts, wf)

