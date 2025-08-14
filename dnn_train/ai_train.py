# dnn_train/train.py

import os # Setting environment variables
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python") # Setting environment variable for Protocol Buffers implementation
import argparse # For parsing command-line arguments
import chess # For chess board representation and operations
import numpy as np 
import tensorflow as tf # For building and training neural networks
from tensorflow.keras import layers, models # For defining neural network layers and models
from tensorflow.keras.callbacks import LambdaCallback # For creating custom callbacks during training
from datasets import load_dataset # For loading datasets from Hugging Face
import pandas as pd # For data manipulation and analysis


print("TF version:", tf.__version__) # Print TensorFlow version
print("GPUs:", tf.config.list_physical_devices("GPU")) # List available GPUs

def parse_args():
    parser = argparse.ArgumentParser()
    # Argument for the model directory, defaulting to an environment variable or a local path
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    return parser.parse_args()
    


def resolve_model_dir(arg_dir: str) -> str:
    """
    Resolves the model directory path. If an S3 URI is provided, 
    it defaults to a local path, otherwise, it uses the provided argument directory.
    """
    local_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    # If user accidentally passed an S3 URI, ignore it and use the local dir
    if not arg_dir or arg_dir.startswith("s3://"): # Check if the argument directory is empty or starts with "s3://"
        if arg_dir and arg_dir.startswith("s3://"):
            print(f"[warn] Ignoring S3 model_dir '{arg_dir}'. Using local '{local_dir}'.") # Warn about ignoring S3 path
        return local_dir # Return the local model directory
    return arg_dir # Return the provided argument directory

CAP_CP = 1000.0 # Cap for centipawn values
INPUT_DIM = 782 # Total input dimension for the neural network (768 board + 1 side to move + 4 castling rights + 8 en-passant file + 1 halfmove clock)
VAL_SIZE = 100_000 # Size of the validation set
BUFFER_SIZE = 150_000 # Buffer size for shuffling the dataset

def board_to_nnue_input(board: chess.Board):
    """Encodes a board position into a 782-element sparse vector."""
    nnue_input = np.zeros(INPUT_DIM, dtype=np.float32)
    for square in chess.SQUARES: # Iterate through all 64 squares on the board
        piece = board.piece_at(square)
        if piece:
            color_offset = 0 if piece.color == chess.WHITE else 384 # Offset for white (0) or black (384) pieces
            piece_offset = (piece.piece_type - 1) * 64 # Offset for piece type (pawn=0, knight=1, etc.)
            square_offset = square # Offset for the square (0-63)
            index = color_offset + piece_offset + square_offset # Calculate the final index in the 768-element vector
            nnue_input[index] = 1 # Set the corresponding index to 1 (one-hot encoding)
            
    # 768: side to move  (+1 white, -1 black)
    nnue_input[768] = 1.0 if board.turn == chess.WHITE else -1.0 # Encode side to move: 1.0 for White, -1.0 for Black

    # 769 - 772: castling rights (WK, WQ, BK, BQ) as 0/1
    nnue_input[769] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0 # White Kingside Castling
    nnue_input[770] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0 # White Queenside Castling
    nnue_input[771] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0 # Black Kingside Castling
    nnue_input[772] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0 # Black Queenside Castling

    # 773 - 780: en-passant file one-hot (a...h) if ep square exists, else zeros
    ep_sq = board.ep_square # Get the en-passant square
    if ep_sq is not None: # If an en-passant square exists
        ep_file = chess.square_file(ep_sq) # Get the file of the en-passant square (0-7)
        nnue_input[773 + ep_file] = 1.0 # Set the corresponding en-passant file index to 1

    # 781: halfmove clock (plies since last capture or pawn push), scaled to [0,1]
    # clip (limit) at 100 to avoid outliers dominating; adjust if you prefer a different cap
    hm = min(board.halfmove_clock, 100) # Clip halfmove clock at 100
    nnue_input[781] = hm / 100.0 # Scale halfmove clock to [0,1]
    return nnue_input

def cp_from_cp_or_mate(cp, mate, cap=CAP_CP):
    # If both centipawn and mate values are None, return None
    if cp is None and mate is None:
        return None
    # If centipawn is None but mate is not, convert mate to a capped centipawn value
    if cp is None:
        # If mate is positive, return the positive cap; otherwise, return the negative cap
        return cap if mate > 0 else -cap
    # If centipawn is available, return it as a float
    return float(cp)

def make_fixed_validation(val_size=VAL_SIZE, buffer_size=BUFFER_SIZE):
    """Pulls a fixed val set from the stream once, converts to (X_val, y_val) numpy arrays."""
    ds = load_dataset("Lichess/chess-position-evaluations", # Load the Lichess chess position evaluations dataset
                      split="train", streaming=True).shuffle(seed=22, buffer_size=buffer_size) # Use the 'train' split, enable streaming, and shuffle with a fixed seed

    X, Y = [], [] # Initialize lists for features (X) and labels (Y)
    for item in ds: # Iterate through each chess position in the dataset
        cp = cp_from_cp_or_mate(item.get("cp"), item.get("mate")) # Get centipawn value or mate score
        fen = item.get("fen") # Get FEN string
        if cp is None or fen is None: # Skip if centipawn or FEN is missing
            continue # Continue to the next item
        try:
            board = chess.Board(fen) # Create a chess board object from FEN
            feats = board_to_nnue_input(board) # Encode the board into NNUE input format
        except Exception: # Catch any exceptions during board creation or encoding
            continue # Skip to the next item if an error occurs

        y_scaled = np.clip(float(cp), -CAP_CP, CAP_CP) / CAP_CP # Scale centipawn value to [-1, 1]
        X.append(feats) # Add features to X list
        Y.append([y_scaled]) # Add scaled label to Y list

        if len(X) >= val_size: # If validation set size is reached
            break

    X = np.asarray(X, dtype=np.float32)                    # (N, INPUT_DIM)
    Y = np.asarray(Y, dtype=np.float32)                    # (N, 1)
    return X, Y


def data_generator(batch_size=1024, buffer_size=BUFFER_SIZE):
    """
    A generator that streams data from Hugging Face, preprocesses it, and yields batches.
    """
    while True:

        dataset = load_dataset("Lichess/chess-position-evaluations", split="train", streaming=True) # Load the dataset in streaming mode
        dataset = dataset.shuffle(seed=np.random.randint(1, 2000), buffer_size=buffer_size) # Shuffle the dataset with a random seed

        # Prefill: advance the iterator to avoid early bias
        it = iter(dataset) # Create an iterator for the dataset
        skipped = 0 # Initialize skipped count
        print(f"[Prefill] Filling shuffle buffer of size {BUFFER_SIZE}...") # Print prefill message
        while skipped < BUFFER_SIZE: # Loop until buffer is filled
            try:
                next(it) # Get the next item from the iterator
                skipped += 1 # Increment skipped count
                if skipped % (BUFFER_SIZE // 10) == 0: # Print progress every 10%
                    pct = (skipped / BUFFER_SIZE) * 100 # Calculate percentage filled
                    print(f"  Prefill progress: {pct:.0f}%") # Print prefill progress
            except StopIteration: # If end of dataset is reached
                break # Exit loop
            except Exception: # Catch any other exceptions
                continue # Continue to the next item
        print("[Prefill] Done. Starting batch generation.") # Print completion message

        X_batch, y_batch = [], [] # Initialize lists for batch features and labels
        
        for item in dataset: # Iterate through the dataset
            try:
                fen = item['fen'] # Get FEN string
                cp = item['cp'] # Get centipawn value
                mate = item.get('mate') # Get mate score
                
                # skip if no label info at all
                if fen is None and cp is None and mate is None: # Skip if no label information is available
                    continue # Continue to the next item

                if cp is None: # If centipawn value is not available
                    if mate is None: # If mate score is also not available
                        continue # Skip to the next item
                    cp = CAP_CP if mate > 0 else -CAP_CP # Assign capped centipawn based on mate score

                board = chess.Board(fen) # Create a chess board object
                nnue_input = board_to_nnue_input(board) # Encode the board
                evaluation = np.clip(float(cp), -CAP_CP, CAP_CP) / CAP_CP # Scale evaluation to [-1, 1]
                    
                X_batch.append(nnue_input) # Add encoded input to batch
                y_batch.append(evaluation) # Add scaled evaluation to batch


                if len(X_batch) == batch_size:
                    X = np.asarray(X_batch, dtype=np.float32)   # (N, INPUT_DIM)
                    y = np.asarray(y_batch, dtype=np.float32)   # (N, 1)
                    yield X, y
                    X_batch, y_batch = [], []                   # Clear batch
            except Exception:
                continue # skip invalid positions



if __name__ == "__main__":
    args = parse_args()
    args.model_dir = resolve_model_dir(args.model_dir) # Resolve the model directory
    os.makedirs(args.model_dir, exist_ok=True) # Create the model directory if it doesn't exist
    X_val, y_val = make_fixed_validation() # Generate fixed validation data
    print("Fixed val shapes:", X_val.shape, y_val.shape) # Print shapes of validation data
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4) # Define Adam optimizer with a constant learning rate
    callbacks = [ # Define callbacks for training
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.9, patience=4, min_delta=1e-3, cooldown=2, min_lr=1e-6, verbose=1), # Reduce learning rate on plateau
        tf.keras.callbacks.ModelCheckpoint(os.path.join(args.model_dir, "best.keras"), save_best_only=True, monitor="val_loss", mode="min"), # Save the best model based on validation loss
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1), # Stop training early if validation loss doesn't improve
        LambdaCallback(
        on_test_begin=lambda logs: print("\n[validation start]"), # Custom callback for validation start
        on_test_end=lambda logs: print("[validation end]\n") # Custom callback for validation end
        )
    ]
    batch_size = 1024 # Batch size for training
    epochs = 20 # Number of training epochs
    steps_per_epoch = 2000 # Number of steps (batches) per epoch
    
    model = tf.keras.Sequential([ # Define the neural network model
        layers.Input(shape=(INPUT_DIM,)), # Input layer with the defined input dimension (782)
        layers.Dense(512, activation="relu"), # First dense layer (ReLu)
        layers.Dropout(0.2), # Dropout
        layers.Dense(256, activation="relu"), # Second dense layer (ReLu)
        layers.Dropout(0.2), # Dropout
        layers.Dense(128, activation="relu"), # Third dense layer (ReLu)
        layers.Dropout(0.2), # Dropout
        layers.Dense(64, activation="relu"), # Fourth dense layer (ReLu)
        layers.Dropout(0.2), # Dropout
        layers.Dense(1, activation="tanh") # Output layer - tanh activation (eval scores b/w -1 and 1)
    ])
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"]) # Compile the model with Adam optimizer, mean squared error loss, and mean absolute error metric
    model.summary() # Print model summary

    print("Starting model training with data from Hugging Face...")

    train_generator = data_generator(batch_size) 

    print("Training dataset created, starting training...")
    # Train the model
    lichess_local_model = model.fit(
        train_generator,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    print("Saving model to:", args.model_dir) # Print message before saving model
    model.save(os.path.join(args.model_dir, "lichess_eval_model.keras")) # Save the trained model

    hist_df = pd.DataFrame(lichess_local_model.history) # Create a DataFrame from training history
    hist_df.to_csv(os.path.join(args.model_dir, "lichess_eval_model.csv"), index=False) # Save training history to CSV
    print("Saved training history to lichess_training_history.csv") # Print message after saving history

    import gc, sys
    # Clear the Keras backend session to free up memory and reset the TensorFlow graph.
    tf.keras.backend.clear_session()

    print("All artifacts saved. Forcing clean exit.")
    os._exit(0)