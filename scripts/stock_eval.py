from stockfish import Stockfish

stockfish = Stockfish(path="stockfish/stockfish-exe", depth=1)

fen_string = "3R4/4P1k1/3p2rp/3N4/5Pp1/6K1/P1r3P1/8 b - - 1 36"
stockfish.set_fen_position(fen_string)
print(stockfish.get_evaluation())
