import chess
import chess.engine

engine = chess.engine.SimpleEngine.popen_uci("stockfish/stockfish-windows-x86-64-avx2.exe")

board = chess.Board()
print(board)
info = engine.analyse(board, chess.engine.Limit(depth=10))
engine.quit()
print("Score:", info["score"].is_mate())