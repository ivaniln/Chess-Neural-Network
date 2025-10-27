# Chess Neural Network — PyTorch Project ♟️

**Short description**  
A neural network trained to evaluate and play chess using self-generated datasets and Stockfish evaluations.  
The main script handles dataset generation, model training, and an interactive console-based chess game where the AI plays against the user.

---

## Repository contents
- `board.py` — main entry point; creates/loads datasets and model, then runs the interactive chess loop  
- `MLP_AI.py` — defines the PyTorch neural network (MLP) and its training logic  
- `Datasets_gen.py` — generates training datasets using Stockfish evaluations  
- `model.pt` — saved model file (created after training)  
- `train_data.pt`, `test_data.pt` — saved datasets (if generated)  

---

## Features
- Uses **Stockfish** to evaluate generated chess positions for supervised learning  
- Builds and trains a **PyTorch MLP** model to predict position values  
- Converts chess boards into tensors for model input  
- Runs an **interactive console game** where the AI evaluates moves and responds  
- Automatically handles dataset generation, saving, and loading  
- GPU support if available (`torch.accelerator`)  

---

## Technologies
- **Python 3**  
- **PyTorch** — neural network architecture and training  
- **Stockfish** — chess engine for position evaluation  
- **python-chess** — board logic, legal moves, FEN management  
- **NumPy** — data manipulation and tensor preprocessing  

---

## How it works
 **Run the main script**
python board.py

On startup, the script will ask:

whether to generate a new dataset or use an existing one

whether to train a new model or load an existing model

Depending on your choice:

If you generate a new dataset, it creates random chess positions, evaluates them with Stockfish, and saves the data.

If you train a new model, it uses the generated dataset and saves model.pt.

After that, an interactive game begins in the console:

You play by typing moves in UCI format (e.g. e2e4).

The AI evaluates each legal move and responds with the one it considers best.

The console shows board state and evaluation values after each move.

## Example usage
Start the project (no additional setup required)
python board.py

Example console flow:
> Do you want to generate a NEW dataset or use EXISTING (n/e): e
> 
> Do you want to learn a NEW model or use EXISTING (n/e): e
> 
(Board printed)

> Your move in uci format: e2e4
> 
> Machines move: d7d5
> 
> Position has value: -0.342

---

## Future improvements
Improve AI move selection using lookahead (e.g. minimax + NN evals)

Add GPU batch inference for speed

Implement reinforcement learning with self-play

Add a simple GUI or connect to lichess-bot API
