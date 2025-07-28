import chess
import random
import chess.engine
import torch

def boardToTensor(board: chess.Board) -> torch.Tensor:
    piece_num = { 'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
         'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10,'k': 11, }
    tensor = torch.zeros(12, 8, 8)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - (square // 8)
            col = square % 8
            idx = piece_num[piece.symbol()]
            tensor[idx, row, col] = 1.0
    return tensor

class ChessDataset(torch.utils.data.Dataset):
    def __init__(self, size = 640):
        super().__init__()
        self.size = size
        self.data = []
        self.x = []
        self.y = []

        print("Creating dataset inputs.")
        while len(self.x) < size*2:
            board = chess.Board()
            for i in range(200):
                self.x.append(board.copy())
                move = random.choice(list(board.legal_moves))
                board.push(move)
                if board.is_game_over():
                    break
                move = random.choice(list(board.legal_moves))
                board.push(move)
                if board.is_game_over():
                    break
        
        print("Creating dataset outputs.")
        self.x = random.choices(self.x, k=size)
        engine = chess.engine.SimpleEngine.popen_uci("stockfish/stockfish-windows-x86-64-avx2.exe")
        for position in self.x:
            info = engine.analyse(position, chess.engine.Limit(depth=10))
            if info["score"].is_mate():
                y_cur = 10
            else:
                y_cur = info["score"].relative.cp / 100
            self.data.append((boardToTensor(position), torch.Tensor([y_cur])))
            self.y.append(y_cur)
        engine.quit()
        print("Dataset created!")

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.data[index]
    
    def save(self, path):
        x1 = []
        for x in self.x:
            x = boardToTensor(x)
            x1.append(x)
        torch.save((x1, list(torch.Tensor(self.y))), path)
    
def load_dataset(path):
    data, targets = torch.load(path)
    dataset = torch.utils.data.TensorDataset(torch.stack(data), torch.stack(targets).unsqueeze(1))
    return dataset

    """
data = ChessDataset(64)
loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)
for x, y in loader:
    print(x)
    print(y)
    """