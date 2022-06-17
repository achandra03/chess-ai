import chess
import chess.pgn

cnt = 0

with open('/Users/arnavchandra/Desktop/chess/data/ficsgamesdb_2021_chess_nomovetimes_252336.pgn') as f:
	game = chess.pgn.read_game(f)
	while(game is not None):
		cnt += 1
		game = chess.pgn.read_game(f)
		print(cnt)

