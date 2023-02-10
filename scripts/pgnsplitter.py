import chess.pgn
import os

pgn = open("data.pgn")
os.chdir('pgns')

first_game = chess.pgn.read_game(pgn)

while first_game: 
    
    game_name = first_game.headers['White'] + '-' + first_game.headers['Black']
    print(game_name)
    out = open(game_name+'.pgn', 'w')
    exporter = chess.pgn.FileExporter(out)
    first_game.accept(exporter)
    first_game = chess.pgn.read_game(pgn)
