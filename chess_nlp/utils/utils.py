from chess import Move, Board
from chess.pgn import Game, read_game
import io
from tqdm import tqdm
import json

def get_word(move: Move) -> str:
    return move.uci()

def get_sentence(game: Game):
    words = [get_word(move) for move in game.mainline_moves()]
    return ' '.join(words)

def read_database(filepath: str) -> list: # each game confined to a single line
    games = []
    with open(filepath, "r") as rf:
        print(f'\nReading data from {filepath} ...')
        for line in tqdm(rf):
            line = line.strip("\n").strip()
            if(line and line[0]=='1'): # that means a new pgn game has started
                games.append(line)
    return games


def get_database_as_sentences(pgn_filepath: str):
    sentences = list()
    vocab = set()

    print(f'\nParsing pgn ...')
    for move_str in tqdm(read_database(pgn_filepath)):
        pgn = io.StringIO(move_str)
        game = read_game(pgn)
        sentences.append(get_sentence(game))
        for word in sentences[-1].split(' '):
            vocab.add(word)
    
    return (sentences, vocab)


def get_next_best_move(board: Board, moves_played: int, unigram_counts: dict, bigram_counts: dict, trigram_counts: dict) -> Move:
    legal_moves = []
    size = moves_played

    last_move = board.pop() if (size >= 1) else None
    second_last_move = board.pop() if (size >= 2) else None
    for move in (second_last_move, last_move):
        if(move): board.push(move)

    for move in board.generate_legal_moves():
        word1 = move.uci()
        if(size == 0):
            # word1
            prob = (unigram_counts.get(word1,0)+1) / (len(unigram_counts)+1)
            legal_moves.append((prob, move))
        elif(size == 1):
            # word2 | word1
            word2 = last_move.uci()
            prob = (bigram_counts.get((word2, word1), 0) + 1) / (unigram_counts.get(word2,0) + 1)
            legal_moves.append((prob, move))
        else:
            # word3 | word2 | word1
            word2 = last_move.uci()
            word3 = second_last_move.uci()
            prob = (trigram_counts.get((word3, word2, word1),0) + 1) / (bigram_counts.get((word3, word2),0) + 1)
            legal_moves.append((prob, move))

    assert(len(legal_moves))

    legal_moves.sort(key=lambda x: x[0], reverse = True)
    return legal_moves[0][1]
