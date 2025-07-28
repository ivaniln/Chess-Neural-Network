import chess
import torch
import MLP_AI
import Datasets_gen

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
    return tensor.to(device)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

model = MLP_AI.MLP().to(device)
model_path = "model.pt"
train_data_path = "train_data.pt"
test_data_path = "test_data.pt"

dataset_choice = input("Do you want to generate a NEW dataset or use EXISTING (n/e): ")

if dataset_choice == "n":
    train_dataset = Datasets_gen.ChessDataset(size=64000)
    test_dataset = Datasets_gen.ChessDataset(size=640)

    train_dataset.save(train_data_path)
    test_dataset.save(test_data_path)
else:
    train_dataset = Datasets_gen.load_dataset("train_data.pt")
    test_dataset = Datasets_gen.load_dataset("test_data.pt")

model_choice = input("Do you want to learn a NEW model or use EXISTING (n/e): ")

if model_choice == "n":
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

    MLP_AI.model_learn(model, train_dataloader, test_dataloader, device, epochs=15, learning_rate=0.001)
    torch.save(model.state_dict(), model_path)
    print("Model created and saved!")
else:
    model.load_state_dict(torch.load(model_path))
    print("Model loaded!")

board = chess.Board()
print(board)

while not board.is_game_over():
    moves = board.legal_moves
    while True:
        move = input("Your move in uci format: ")
        try:
            if chess.Move.from_uci(f"{move}") in moves:
                board.push(chess.Move.from_uci(f"{move}"))
                break
            else:
                print("Illigal move")
        except:
            print("Illigal move")
    
    if board.is_game_over():
        break

    moves = board.legal_moves
    best_move = chess.Move.from_uci("e2e4")
    best_val = float("inf")
    for move in moves:
        board.push(move)
        board_tensor = boardToTensor(board).unsqueeze(0)
        val = model(board_tensor)
        if val < best_val:
            best_val = val
            best_move = move
        board.pop()
    board.push(best_move)
    print(f"Position has value: {best_val.__float__()}")
    print(f"Machines move: {best_move}")
    print(board)

print(f"Checkmate: {board.is_checkmate()}")