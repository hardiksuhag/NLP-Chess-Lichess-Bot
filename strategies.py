"""
Some example strategies for people who want to create a custom, homemade bot.
And some handy classes to extend
"""

from chess.engine import PlayResult
import chess
import random
from engine_wrapper import EngineWrapper
import pickle

class FillerEngine:
    """
    Not meant to be an actual engine.

    This is only used to provide the property "self.engine"
    in "MinimalEngine" which extends "EngineWrapper"
    """
    def __init__(self, main_engine, name=None):
        self.id = {
            "name": name
        }
        self.name = name
        self.main_engine = main_engine

    def __getattr__(self, method_name):
        main_engine = self.main_engine

        def method(*args, **kwargs):
            nonlocal main_engine
            nonlocal method_name
            return main_engine.notify(method_name, *args, **kwargs)

        return method


class MinimalEngine(EngineWrapper):
    """
    Subclass this to prevent a few random errors

    Even though MinimalEngine extends EngineWrapper,
    you don't have to actually wrap an engine.

    At minimum, just implement `search`,
    however you can also change other methods like
    `notify`, `first_search`, `get_time_control`, etc.
    """
    def __init__(self, commands, options, stderr, draw_or_resign, name=None, **popen_args):
        super().__init__(options, draw_or_resign)

        self.engine_name = self.__class__.__name__ if name is None else name

        self.engine = FillerEngine(self, name=self.name)
        self.engine.id = {
            "name": self.engine_name
        }

    def search(self, board, time_limit, ponder, draw_offered, root_moves):
        """
        The method to be implemented in your homemade engine

        NOTE: This method must return an instance of "chess.engine.PlayResult"
        """
        raise NotImplementedError("The search method is not implemented")

    def notify(self, method_name, *args, **kwargs):
        """
        The EngineWrapper class sometimes calls methods on "self.engine".
        "self.engine" is a filler property that notifies <self>
        whenever an attribute is called.

        Nothing happens unless the main engine does something.

        Simply put, the following code is equivalent
        self.engine.<method_name>(<*args>, <**kwargs>)
        self.notify(<method_name>, <*args>, <**kwargs>)
        """
        pass


class ExampleEngine(MinimalEngine):
    pass


# Strategy names and ideas from tom7's excellent eloWorld video

class RandomMove(ExampleEngine):
    def search(self, board, *args):
        return PlayResult(random.choice(list(board.legal_moves)), None)


class Alphabetical(ExampleEngine):
    def search(self, board, *args):
        moves = list(board.legal_moves)
        moves.sort(key=board.san)
        return PlayResult(moves[0], None)


class FirstMove(ExampleEngine):
    """Gets the first move when sorted by uci representation"""
    def search(self, board, *args):
        moves = list(board.legal_moves)
        moves.sort(key=str)
        return PlayResult(moves[0], None)

class Bigram(ExampleEngine):

    def count_moves(self, board):
        temp = board.copy()
        num_moves = 0
        while True:
            try:
                temp.pop()
                num_moves += 1
            except:
                break
        return num_moves

    def get_next_best_move(self, board_orig):
        board = board_orig.copy()
        moves_played = self.count_moves(board)

        try:
            unigram_counts = self.unigram_counts
            bigram_counts = self.bigram_counts
            trigram_counts = self.trigram_counts
        except:
            with open("chess_nlp/bigram/unigram_counts.pkl", "rb") as rf:
                self.unigram_counts = pickle.load(rf)

            with open("chess_nlp/bigram/bigram_counts.pkl", "rb") as rf:
                self.bigram_counts = pickle.load(rf)

            with open("chess_nlp/bigram/trigram_counts.pkl", "rb") as rf:
                self.trigram_counts = pickle.load(rf)

        unigram_counts = self.unigram_counts
        bigram_counts = self.bigram_counts
        trigram_counts = self.trigram_counts

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

    def search(self, board, *args):
        return PlayResult(self.get_next_best_move(board), None)


class MiniMax(ExampleEngine):

    def minimaxRoot(self, depth, board,isMaximizing):
        possibleMoves = board.legal_moves
        bestMove = -9999
        secondBest = -9999
        thirdBest = -9999
        bestMoveFinal = None
        for move in possibleMoves:
            board.push(move)
            value = max(bestMove, self.minimax(depth - 1, board, not isMaximizing))
            board.pop()
            if( value > bestMove):
                thirdBest = secondBest
                secondBest = bestMove
                bestMove = value
                bestMoveFinal = move
        return bestMoveFinal

    def minimax(self, depth, board, is_maximizing):
        if(depth == 0):
            return -self.evaluation(board)
        possibleMoves = board.legal_moves
        if(is_maximizing):
            bestMove = -9999
            for move in possibleMoves:
                board.push(move)
                bestMove = max(bestMove, self.minimax(depth - 1, board, not is_maximizing))
                board.pop()
            return bestMove
        else:
            bestMove = 9999
            for move in possibleMoves:
                board.push(move)
                bestMove = min(bestMove, self.minimax(depth - 1, board, not is_maximizing))
                board.pop()
            return bestMove

    def evaluation(self, board):
        i = 0
        evaluation = 0
        x = True
        try:
            x = bool(board.piece_at(i).color)
        except AttributeError as e:
            x = x
        while i < 63:
            i += 1
            evaluation = evaluation + (self.getPieceValue(str(board.piece_at(i))) if x else -self.getPieceValue(str(board.piece_at(i))))
        return evaluation

    def getPieceValue(self, piece):
        if(piece == None):
            return 0
        value = 0
        if piece == "P" or piece == "p":
            value = 10
        if piece == "N" or piece == "n":
            value = 30
        if piece == "B" or piece == "b":
            value = 30
        if piece == "R" or piece == "r":
            value = 50
        if piece == "Q" or piece == "q":
            value = 90
        if piece == 'K' or piece == 'k':
            value = 900
        #value = value if (board.piece_at(place)).color else -value
        return value

    def get_next_best_move(self, board):
        temp = board.copy()
        return self.minimaxRoot(4, temp, not board.turn)

    def search(self, board, *args):
        return PlayResult(self.get_next_best_move(board), None)


class AlphaBetaPruning(ExampleEngine):

    def evaluate_piece(self, piece, square) :

        pawnEvalWhite = [
            0, 0, 0, 0, 0, 0, 0, 0,
            5, 10, 10, -20, -20, 10, 10, 5,
            5, -5, -10, 0, 0, -10, -5, 5,
            0, 0, 0, 20, 20, 0, 0, 0,
            5, 5, 10, 25, 25, 10, 5, 5,
            10, 10, 20, 30, 30, 20, 10, 10,
            50, 50, 50, 50, 50, 50, 50, 50,
            0, 0, 0, 0, 0, 0, 0, 0
        ]
        pawnEvalBlack = list(reversed(pawnEvalWhite))

        knightEval = [
            -50, -40, -30, -30, -30, -30, -40, -50,
            -40, -20, 0, 0, 0, 0, -20, -40,
            -30, 0, 10, 15, 15, 10, 0, -30,
            -30, 5, 15, 20, 20, 15, 5, -30,
            -30, 0, 15, 20, 20, 15, 0, -30,
            -30, 5, 10, 15, 15, 10, 5, -30,
            -40, -20, 0, 5, 5, 0, -20, -40,
            -50, -40, -30, -30, -30, -30, -40, -50
        ]

        bishopEvalWhite = [
            -20, -10, -10, -10, -10, -10, -10, -20,
            -10, 5, 0, 0, 0, 0, 5, -10,
            -10, 10, 10, 10, 10, 10, 10, -10,
            -10, 0, 10, 10, 10, 10, 0, -10,
            -10, 5, 5, 10, 10, 5, 5, -10,
            -10, 0, 5, 10, 10, 5, 0, -10,
            -10, 0, 0, 0, 0, 0, 0, -10,
            -20, -10, -10, -10, -10, -10, -10, -20
        ]
        bishopEvalBlack = list(reversed(bishopEvalWhite))

        rookEvalWhite = [
            0, 0, 0, 5, 5, 0, 0, 0,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            5, 10, 10, 10, 10, 10, 10, 5,
            0, 0, 0, 0, 0, 0, 0, 0
        ]
        rookEvalBlack = list(reversed(rookEvalWhite))

        queenEval = [
            -20, -10, -10, -5, -5, -10, -10, -20,
            -10, 0, 0, 0, 0, 0, 0, -10,
            -10, 0, 5, 5, 5, 5, 0, -10,
            -5, 0, 5, 5, 5, 5, 0, -5,
            0, 0, 5, 5, 5, 5, 0, -5,
            -10, 5, 5, 5, 5, 5, 0, -10,
            -10, 0, 5, 0, 0, 0, 0, -10,
            -20, -10, -10, -5, -5, -10, -10, -20
        ]

        kingEvalWhite = [
            20, 30, 10, 0, 0, 10, 30, 20,
            20, 20, 0, 0, 0, 0, 20, 20,
            -10, -20, -20, -20, -20, -20, -20, -10,
            20, -30, -30, -40, -40, -30, -30, -20,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30
        ]
        kingEvalBlack = list(reversed(kingEvalWhite))


        piece_type = piece.piece_type
        mapping = []
        if piece_type == chess.PAWN:
            mapping = pawnEvalWhite if piece.color == chess.WHITE else pawnEvalBlack
        if piece_type == chess.KNIGHT:
            mapping = knightEval
        if piece_type == chess.BISHOP:
            mapping = bishopEvalWhite if piece.color == chess.WHITE else bishopEvalBlack
        if piece_type == chess.ROOK:
            mapping = rookEvalWhite if piece.color == chess.WHITE else rookEvalBlack
        if piece_type == chess.QUEEN:
            mapping = queenEval
        if piece_type == chess.KING:
            mapping = kingEvalWhite if piece.color == chess.WHITE else kingEvalBlack

        return mapping[square]


    def calc_board_value(self, board):
        piece_value = {
            chess.PAWN: 100,
            chess.ROOK: 500,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.QUEEN: 900,
            chess.KING: 20000
        }

        total = 0

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if not piece:
                continue

            value = piece_value[piece.piece_type] + self.evaluate_piece(piece, square)
            total += value if piece.color == chess.WHITE else -value

        return total


    def evaluate_position(self, board: chess.Board, player: chess.Color):
        if board.is_checkmate():
            if player == chess.WHITE:
                return -float("inf")
            else:
                return float("inf")
        if board.is_stalemate():
            return 0.0
        else:
            return self.calc_board_value(board)

    def alpha_beta_search(self, board, depth = 5):
        if board.turn == chess.WHITE:
            value, move = self.max_value(board, depth - 1, alpha=-float("inf"), beta=float("inf"))
        else:
            value, move = self.min_value(board, depth - 1, alpha=-float("inf"), beta=float("inf"))
        return move

    def max_value(self, board, depth, alpha, beta):
        if depth == 0:
            value = self.evaluate_position(board, board.turn)
            return value, None
        v = -float("inf")
        move = None
        for action in board.legal_moves:
            board.push(action)
            v2, _ = self.min_value(board, depth - 1, alpha, beta)
            board.pop()
            if v2 > v:
                v, move = v2, action
                alpha = max(alpha, v)
            if alpha >= beta:
                return v, move
        return v, move

    def min_value(self, board, depth, alpha, beta):
        if depth == 0:
            value = self.evaluate_position(board, board.turn)
            return value, None
        v = float("inf")
        move = None
        for action in board.legal_moves:
            board.push(action)
            v2, _ = self.max_value(board, depth - 1, alpha, beta)
            board.pop()
            if v2 < v:
                v, move = v2, action
                beta = min(beta, v)
            if beta <= alpha:
                return v, move
        return v, move


    def search(self, board, *args):
        return PlayResult(self.alpha_beta_search(board), None)    
