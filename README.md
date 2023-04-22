# BetaChess: MCTS-based Chess Engine with Deep Neural Networks

This project uses Monte Carlo Tree Search (MCTS) and a deep learning model to create a chess AI. The code is organized into four main Python scripts: `beta_chess.py`, `MCTS.py`, `pipeline.py`, and `encoding.py`.

## Requirements

- Python 3.6+
- PyTorch
- NumPy
- Matplotlib
- python-chess

## beta_chess.py

`beta_chess.py` contains the following classes and functions:

- `BoardData`: A custom dataset class for chess board positions.
- `ConvBlock`, `ResBlock`, `OutBlock`: Neural network building blocks.
- `ChessNet`: The main neural network architecture.
- `ChessLoss`: Custom loss function for training the network.
- `train()`: Function for training the neural network.

These are inspired by the AlphaZero paper and take a similar approach to the neural network architecture and training.

## MCTS.py

`MCTS.py` is a Python script that contains a customized implementation of the Monte Carlo Tree Search algorithm. It includes the following key features:

- A more efficient implementation of MCTS with a focus on memory usage reduction.
- Caching legal moves to speed up the search process.
- Incorporation of Dirichlet noise for better exploration in the root node.
- A custom backup function that considers the game turn to adjust the value estimate accordingly.
- Support for multiprocessing during the self-play phase.
- Logging and tracking game statistics for analysis.
- Optimizations in both execution speed and move exploration like:
    - Parent-Q initialization
    - Virtual loss
    - Sub-tree reuse
    - Parallelization
    - Predictor + Upper Confidence bounds applied to Trees (PUCT)
    
## encoding.py
`encoding.py` only contains helper functions to encode and decode the board and any move on the board.

## pipeline.py

`pipeline.py` is a Python script that combines the chess AI training and Monte Carlo Tree Search (MCTS) self-play into a single pipeline. 
It simply contains 2 functions:

- `run_MCTS()`: Function to run MCTS self-play for a given iteration.
- `run_net_training()`: Function to train the neural network for a given iteration.

To run the pipeline, execute the `pipeline.py` script. It will create the necessary directories if they don't exist and run the MCTS and neural network training for the specified number of iterations.

## Running the code

1. Install the required dependencies.

```bash
pip install -r requirements.txt
```

2. Execute the `pipeline.py` script to run the entire pipeline with the specified number of iterations.

```bash
python pipeline.py
```

This will run the MCTS self-play and neural network training for the specified number of iterations.

## Additional Notes

- The MCTS and neural network training hyperparameters can be modified in the `pipeline.py` script.
- The MCTS self-play and neural network training are parallelized using PyTorch's multiprocessing module. Adjust the `NUM_PROCESSES` parameter in the `pipeline.py` script based on your system's capabilities.
