"""
This file contains the chess AI's core logic, including the Negamax search algorithm
with alpha-beta pruning and a neural network-based evaluation function.
"""
import numpy as np # for numeric array operation (encoding)
import tensorflow as tf # for loading and running the neural network model
import random # for random move selection in fallback logic
import sys

# Constants for evaluation
CHECKMATE_SCORE = 1000 # score assigned to checkmate positions
STALEMATE_SCORE = 0 # score assigned to stalemate positions
MAX_DEPTH = 4 # search depth for negamax
MODEL_PATH = "lichess_eval_model.keras" # filepath to trained keras model
MAX_LEAF_BATCH = 100 # batched leaf evaluation batch size (legal moves per node mostly < 100, 218 theoretical max)

# Weights for evaluation components, extra features can be added later (example: mobility, king safety, etc.)
WEIGHTS = {
    "dnn": 1.0, # main trust in neural network evaluation
}

# Global placeholders set during run_ai_loop
eval_model = None # TensorFlow model for evaluation
MODEL_INPUT = None # Preallocated input buffer for the model (1 × 768)
prediction_graph = None # Compiled prediction graph for speed (if available)

# maps piece codes to channel indices 0–11
piece_to_index = {
    "wp":0, "wN":1, "wB":2,
    "wR":3, "wQ":4, "wK":5,
    "bp":6, "bN":7, "bB":8,
    "bR":9, "bQ":10,"bK":11
}

'''
Vectorized DNN eval evaluation
This is a 768-vector encoding of the chess board, where each piece type has its
'''
def eval_batch(encoded_boards):
    input_array = np.asarray(encoded_boards, dtype=np.float32) # Convert to numpy array
    if input_array.ndim == 1: # if a single board is provided, reshape to 2D
        input_array = input_array[np.newaxis, :] # reshape to 1x768
    if prediction_graph is not None: # use compiled graph for speed
        predictions = prediction_graph(input_array) # run the model on the input array
    else: # fallback to eager execution
        predictions = eval_model(input_array, training=False) # run the model on the input array
    predictions = np.asarray(predictions) # Convert predictions to numpy array
    if predictions.ndim == 2: # if predictions are 2D, take the first column (single output per board)
        predictions = predictions[:,0] # take the first column
    return predictions # return the predictions as a 1D array

'''
Piece values for evaluation
These values are used to evaluate the material on the board.
'''
def square_index(r, c): # flatten (row,col) i.e. 0 to 63
    return r*8 + c

'''
One-hot encode the board into a 768-vector for efficiently updatable neural network input.
This builds a fresh encoding from scratch.
Used at root node of search or when encoding don't already exist for modification
'''
def encode_board_nn(board):
    encoded_vector = np.zeros(768, dtype=np.float32) # initialize 12×64 array as flat vector
    for row_index in range(8): # iterate over rows
        for col_index in range(8): # iterate over columns
            piece = board[row_index][col_index] # piece at (row,col)
            if piece != "--": # If the position on the board is not empty
                encoded_vector[piece_to_index[piece] * 64 + square_index(row_index, col_index)] = 1.0 # update encoding
    return encoded_vector # return the encoding

'''
Lightweight move property helpers for NNUE encoding.
These avoid calling chessEngine methods directly for performance reasons.
During deep search the AI simulates thousands of positions, so repeatedly
asking the full engine to check captures, promotions etc. would be too slow.
Instead, we read and interpret the relevant data directly from the Move object
'''
# Helper to map promotion choice to final piece code (example, 'Q' to 'wQ'/'bQ')
def promotion_piece_code(piece_from, promotion_choice):
    if promotion_choice is None: # if no promotion choice, return the original piece code
        return piece_from # piece code at the start square (e.g., 'wP' or 'bP')
    # choice could be 'Q','R','B','N' or full code like 'wQ'/'bQ'
    if isinstance(promotion_choice, str) and len(promotion_choice) == 2 and promotion_choice[0] in ("w","b"): # if choice is a full piece code
        return promotion_choice # return the full piece code
    piece_color = piece_from[0] # 'w' or 'b' # get the color of the piece
    piece_letter = str(promotion_choice).upper() # convert choice to uppercase letter
    if piece_letter not in ("Q","R","B","N"): # if choice is not a valid piece type
        piece_letter = "Q" # default to queen
    return f"{piece_color}{piece_letter}" # return the piece code with color and type (e.g., 'wQ' or 'bQ')

'''
Record prior value so we can undo later.
'''
def undo_log(encoding_vector, index, new_value, changes_to_undo):
    previous_value = encoding_vector[index] # store the previous value at the index
    if previous_value == new_value: # if the value is already set to the new value, do nothing
        return
    changes_to_undo.append((index, previous_value)) # record the change for undo
    encoding_vector[index] = new_value

'''
Annotate each move for coding
This function is a lightweight helper to prepare a Move object for encoding updates.
It populates necessary attributes by using information already available on the Move object from the engine.
This removes the redundancy of re-calculating move properties.
'''
def annotate_move_for_encoding(game_state, move):
    # Use move attributes directly from the engine's Move object
    piece_from = move.moved_piece
    captured_piece = move.captured_piece
    is_en_passant = move.is_en_passant
    is_castling = move.is_castling_move

    # Determine captured piece position for en passant since it's not the end square
    if is_en_passant:
        captured_piece_code = "bp" if piece_from[0] == "w" else "wp"
        captured_row, captured_col = (move.end_row + 1, move.end_col) if piece_from[0] == "w" else (move.end_row - 1, move.end_col)
    else:
        captured_piece_code = captured_piece
        captured_row, captured_col = move.end_row, move.end_col
    
    # Handle promotion to get the correct final piece code
    piece_to = promotion_piece_code(piece_from, getattr(move, "promotion_choice", None))

    # Store these annotations as attributes on the move object for easy access later
    move._nn_pf = piece_from
    move._nn_pc = captured_piece_code
    move._nn_pt = piece_to
    move._nn_cap_r = captured_row
    move._nn_cap_c = captured_col
    move._nn_is_castling = is_castling

'''
Update an existing encoding to reflect a single move without recomputing from scratch.
At all deeper nodes, we already encode parent position.
This is called before game_state.make_move(move).
'''
def apply_encoding(encoding_vector, move, changes_to_undo):
    # derive basic squares from the move object
    start_row, start_col = move.start_row, move.start_col
    end_row, end_col = move.end_row, move.end_col

    # Use pre-annotated move attributes
    moved_piece_code = move._nn_pf
    destination_piece_code = move._nn_pt
    captured_piece_code = move._nn_pc

    # Handle the moved piece at its origin (this piece is now gone from its original square)
    start_index = piece_to_index[moved_piece_code] * 64 + square_index(start_row, start_col)
    undo_log(encoding_vector, start_index, 0.0, changes_to_undo)

    # Handle the captured piece (removed from the board)
    if captured_piece_code != "--" and captured_piece_code is not None:
        captured_row = move._nn_cap_r
        captured_col = move._nn_cap_c
        captured_index = piece_to_index[captured_piece_code] * 64 + square_index(captured_row, captured_col)
        undo_log(encoding_vector, captured_index, 0.0, changes_to_undo)

    # Handle the piece at the destination square after the move
    end_index = piece_to_index[destination_piece_code] * 64 + square_index(end_row, end_col)
    undo_log(encoding_vector, end_index, 1.0, changes_to_undo)

    # Handle the rook's move during a castling
    if move._nn_is_castling:
        rook_code = "wR" if moved_piece_code == "wK" else "bR"
        if end_col > start_col: # Kingside castling
            rook_start_col, rook_end_col = 7, 5
        else: # Queenside castling
            rook_start_col, rook_end_col = 0, 3

        rook_row = start_row
        # Clear the rook from its original position.
        rook_start_index = piece_to_index[rook_code] * 64 + square_index(rook_row, rook_start_col)
        undo_log(encoding_vector, rook_start_index, 0.0, changes_to_undo)
        # Place the rook at its new position.
        rook_end_index = piece_to_index[rook_code] * 64 + square_index(rook_row, rook_end_col)
        undo_log(encoding_vector, rook_end_index, 1.0, changes_to_undo)


'''
Undo the last apply_encoding by restoring the board to its
previous state after making a temporary move during the AI's search (uses record from undo_log).
'''
def undo_encoding(encoding_vector, changes_to_undo):
    while changes_to_undo: # Loop as long as there are changes to undo
        index, previous_value = changes_to_undo.pop() # Remove the last change which is a tuple of (index, previous_value)
        encoding_vector[index] = previous_value # Restore the previous value to the specified index in the encoding vector

'''
Evaluation score of the board based on terminal node cases
Here positive score is good for white and a negative score is good for black.
'''
def terminal_eval_score(game_state):
    if game_state.is_checkmate: # If the game is checkmate
        if game_state.white_to_move: # If it's white's turn and the game is checkmate
            return -CHECKMATE_SCORE # black wins
        else: # If it's black's turn and the game is checkmate
            return CHECKMATE_SCORE # white wins
    elif game_state.is_stalemate: # If the game is in stalemate
        return STALEMATE_SCORE
    else:
        raise RuntimeError("terminal_eval_score called on non-terminal position")

'''
This is the entry point for the background AI process.
It loads the neural network model once (for efficiency) and then waits
in a loop for game states from the main process via the input queue.
For each request, it calculates the best move and sends it back via the output queue.
'''
def run_ai_loop(input_queue, output_queue):
    global eval_model, MODEL_INPUT, prediction_graph # declare globals so the model and buffers are accessible in other AI functions
    eval_model = tf.keras.models.load_model(MODEL_PATH, compile=False) # Load pre-trained Keras model from disk (compiled=False skips training setup)
    MODEL_INPUT = np.zeros((1, 768), dtype=np.float32) # Preallocate a NumPy array for the model input (1 position × 768 features), # This avoids allocating a new array for every evaluation call

    # Define an optional tf.function wrapper to compile the model call for speed
    @tf.function(jit_compile=False) # Convert to a TensorFlow graph for faster repeated calls
    def build_prediction_graph(model):
        # Return a function that runs the model in inference mode
        def predict_fn(x):
            return model(x, training=False) # create callable function for the model once
        return predict_fn
    try:
        prediction_graph = build_prediction_graph(eval_model) # build and compile the prediction graph for faster inference (predict() not used here)
        _ = prediction_graph(MODEL_INPUT) # Run one dummy prediction to compile and trigger graph building once
    except Exception: # If compilation fails, fall back to normal eager execution
        prediction_graph = None

    while True: # Main loop - keep processing AI move requests from the main process
        # Get the next request from the main process
        # pos_key -> move number at time of request (for stale-move detection)
        pos_key, game_state, legal_moves = input_queue.get()
        best_move = get_best_move(game_state, legal_moves) # Use AI search to select the best move from the current position
        output_queue.put((pos_key, best_move)) # Send the result back to the main process along with the position key

'''
Helper function to make first recursive call to best move function
'''
def get_best_move(game_state, legal_moves):
    global best_moves_list, function_calls # Initialize the empty list of best moves, next best move to None and function calls to 0
    best_moves_list = [] # Initialize the list of best moves
    function_calls = 0 # Initialize the function calls to 0
    root_enc = encode_board_nn(game_state.board) # encode once at root
    turn_multiplier = 1 if game_state.white_to_move else -1 # Determine if it's white's turn
    random.shuffle(legal_moves) # Shuffle the legal moves to add randomness
    negamax_alpha_beta_search(game_state, legal_moves, MAX_DEPTH, -CHECKMATE_SCORE, CHECKMATE_SCORE, turn_multiplier, 0, encoding=root_enc)
    # print(f"Number of function calls made: {function_calls}") # check performance of the algorithm
    if best_moves_list: # If there are best moves found
        return random.choice(best_moves_list) # Randomly select one of the best moves for breaking ties
    else: # If no best moves found
        return random_AI_move(legal_moves)

'''
Negamax algorithm to find the best move (recursive function) with alpha-beta pruning (improved efficiency)
'''
def negamax_alpha_beta_search(game_state, legal_moves, search_depth_left, alpha, beta,
                                     turn_multiplier, depth_from_root, encoding):
    global best_moves_list, function_calls # Initialize the empty best moves list, next best move to None and function calls to 0
    function_calls += 1 # Increment the function calls count
    if not legal_moves: # if no more legal moves
        return turn_multiplier * terminal_eval_score(game_state) # Evaluate the board score for the current game state and return it

    # When search_depth_left == 1 i.e., batched leaf evaluation (batching per node at depth-1, not all leaves of the whole tree)
    if search_depth_left == 1: # if we are at the second last depth of our search
        max_score = -CHECKMATE_SCORE # set the max score to the lowest possible value
        i = 0 # i is batch start index
        while i < len(legal_moves): # more moves remain in next batch (incase >100 legal moves exist and pruning did not occur)
            moves_batch = legal_moves[i:i + MAX_LEAF_BATCH] # take a batch of moves to evaluate
            encoded_boards_to_evaluate = [] # list to hold encoded boards for batch evaluation
            batch_index_map = [] # list to map evaluation results back to the original move index
            move_scores_raw = [None] * len(moves_batch) # list to store the raw scores of each move in the batch
            for j, move in enumerate(moves_batch): # j is batch move index
                new_enc = encoding.copy() # create a copy of the parent encoding to apply changes to
                annotate_move_for_encoding(game_state, move) # minimal annotations for nn board
                changes_to_undo = [] # initialize a list to store changes for quick undo
                apply_encoding(new_enc, move, changes_to_undo) # apply the move to the encoding
                game_state.make_move(move) # make the move on the actual game state
                next_legal_moves = game_state.get_valid_moves() # get the legal moves from the new position
                if (len(next_legal_moves) == 0) and (game_state.is_checkmate or game_state.is_stalemate): # If child is terminal (checkmate or stalemate) avoid NN call
                    move_scores_raw[j] = terminal_eval_score(game_state) # directly evaluate the terminal state without a neural network call
                else:
                    encoded_boards_to_evaluate.append(new_enc) # add the new encoding to the batch for evaluation
                    batch_index_map.append(j) # record the index to map the score back
                game_state.undo_move() # undo the move on the actual game state
                undo_encoding(new_enc, changes_to_undo)
            if encoded_boards_to_evaluate: # if there are boards to evaluate
                preds = eval_batch(encoded_boards_to_evaluate) # evaluate the batch of boards at once
                for eval_index, j in enumerate(batch_index_map): # loop through the evaluation results
                    move_scores_raw[j] = WEIGHTS["dnn"] * float(preds[eval_index]) # store the final weighted score
            for j, move in enumerate(moves_batch):
                score = turn_multiplier * move_scores_raw[j] # calculate the score for the current player's perspective
                # negamax check
                if score > max_score: # If the score is greater than the current max score
                    max_score = score # Update the maximum score
                    if search_depth_left == MAX_DEPTH: # If the search depth is at the maximum depth
                        best_moves_list = [move] # Initialize the list of best moves
                elif score == max_score and search_depth_left == MAX_DEPTH: # If the score is equal to the current max score (tie)
                    best_moves_list.append(move) # Add the move to the list of best moves
                # tracking alpha/beta for parent level pruning (between second last nodes)
                if max_score > alpha:
                    alpha = max_score # Update alpha to the maximum score found
                if alpha >= beta: # If alpha is greater than or equal to beta
                    return max_score # exit the loop
            i += MAX_LEAF_BATCH # move to the next batch of moves
        return max_score # return the best score found in this branch
    # When search_depth_left >= 2
    max_score = -CHECKMATE_SCORE # Initialize to a very low score for the player's best move (max score obtainable)
    for move in legal_moves: # loop through each legal move the player can make
        new_enc = encoding.copy() # copy encoding (fast for 768 floats)
        annotate_move_for_encoding(game_state, move)
        changes_to_undo = [] # initialize the changes stack
        apply_encoding(new_enc, move, changes_to_undo) # apply encoding changes before mutating the game state

        game_state.make_move(move) # Make the player's move in the game state
        next_possible_moves = game_state.get_valid_moves() # Get the valid moves for the next possible moves
        score = -negamax_alpha_beta_search(game_state, next_possible_moves, search_depth_left - 1, -beta, -alpha, -turn_multiplier, depth_from_root + 1, new_enc) # Recursive call to find the best move for the opponent
        game_state.undo_move() # Undo the player's move to evaluate the next one
        undo_encoding(new_enc, changes_to_undo) # restore encoding quickly
        # negamax check
        if score > max_score: # If the score is greater than the current max score
            max_score = score # Update the maximum score
            if search_depth_left == MAX_DEPTH: # If the search depth is at the maximum depth
                best_moves_list = [move] # Initialize the list of best moves
        elif score == max_score: # If the score is equal to the current max score (tie)
            if search_depth_left == MAX_DEPTH: # If the search depth is at the maximum depth
                best_moves_list.append(move) # Add the move to the list of best moves
        # pruning happens here
        if max_score > alpha:
            alpha = max_score # Update alpha to the maximum score found
        if alpha >= beta: # If alpha is greater than or equal to beta
            break # exit the loop (pruning)
    return max_score # Return the maximum score found

'''
Backup random AI move if AI negamax algorithm fails or returns None.
'''
def random_AI_move(legal_moves):
    return legal_moves[random.randint(0, len(legal_moves) - 1)] # Random AI move selection