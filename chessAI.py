"""
This file contains the chess AI's core logic, including the Negamax search algorithm
with alpha-beta pruning and a neural network-based evaluation function.
"""
import numpy as np # for numeric array operation (encoding)
import tensorflow as tf # for loading and running the neural network model
import random # for random move selection in fallback logic
from collections import defaultdict # for using killer moves' history tables

# Constants for evaluation
CHECKMATE_SCORE = 1000 # score assigned to checkmate positions
STALEMATE_SCORE = 0 # score assigned to stalemate positions
MAX_DEPTH = 4 # search depth for negamax
MODEL_PATH = "lichess_eval_model.keras" # filepath to trained keras model
MAX_LEAF_BATCH = 16 # batched leaf evaluation batch size (legal moves per node mostly < 100)

# Weights for evaluation components, extra features can be added later (example: mobility, king safety, etc.)
WEIGHTS = {
    "dnn": 1.0, # main trust in neural network evaluation
}

# Global placeholders set during run_ai_loop
eval_model = None # TensorFlow model for evaluation
MODEL_INPUT = None # Preallocated input buffer for the model (1 × 768)
prediction_graph = None # Compiled prediction graph for speed (if available)

# A map to convert piece characters to a 0-5 integer index.
PIECE_TYPE_MAP = {'p': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5}

# transposition table
TRANSPOSITION_TABLE = {} # global dictionary for storing board states and their evaluation scores
EXACT, LOWER, UPPER = 0, 1, 2 # transposition flags to classify the stored score as exact, a lower bound, or an upper bound

# Killer moves for each ply (search depth)
# History of good moves (large table for long-term storage)
KILLER_MOVES = defaultdict(lambda: [None, None]) # two killers per ply (auto-creates [None, None] for unseen ply), stores non-capture moves that caused a beta cutoff at a given depth
HISTORY_MOVES = defaultdict(int) # MOVES: stores a score for each move, incremented when the move causes a cutoff

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
    encoded_vector = np.zeros(768, dtype=np.float32)
    for row_index in range(8):
        for col_index in range(8):
            piece = board[row_index][col_index]
            if piece != "--":
                color_offset = 0 if piece[0] == 'w' else 384
                piece_type_char = piece[1]
                # Calculate the piece offset using the global map
                piece_offset = PIECE_TYPE_MAP[piece_type_char] * 64
                # The square offset is the same as before.
                square_offset = square_index(row_index, col_index)
                # Combine the offsets to get the final index
                index = color_offset + piece_offset + square_offset
                encoded_vector[index] = 1
    return encoded_vector

'''
Lightweight move property helpers for nn encoding.
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
def apply_encoding(encoding_vector, move):
    # derive basic squares from the move object
    start_row, start_col = move.start_row, move.start_col
    end_row, end_col = move.end_row, move.end_col

    # Use pre-annotated move attributes
    moved_piece_code = move._nn_pf
    destination_piece_code = move._nn_pt
    captured_piece_code = move._nn_pc
    
    # Helper function to calculate the nn index
    def get_nn_index(piece_code, row, col):
        color_offset = 0 if piece_code[0] == 'w' else 384
        piece_type_char = piece_code[1]
        piece_offset = PIECE_TYPE_MAP[piece_type_char] * 64
        square_offset = square_index(row, col)
        return color_offset + piece_offset + square_offset

    # Handle the moved piece at its origin (this piece is now gone from its original square)
    start_index = get_nn_index(moved_piece_code, start_row, start_col)
    encoding_vector[start_index] = 0.0

    # Handle the captured piece (removed from the board)
    if captured_piece_code != "--" and captured_piece_code is not None:
        captured_row = move._nn_cap_r
        captured_col = move._nn_cap_c
        captured_index = get_nn_index(captured_piece_code, captured_row, captured_col)
        encoding_vector[captured_index] = 0.0

    # Handle the piece at the destination square after the move
    end_index = get_nn_index(destination_piece_code, end_row, end_col)
    encoding_vector[end_index] = 1.0

    # Handle the rook's move during a castling
    if move._nn_is_castling:
        rook_code = "wR" if moved_piece_code == "wK" else "bR"
        if end_col > start_col: # Kingside castling
            rook_start_col, rook_end_col = 7, 5
        else: # Queenside castling
            rook_start_col, rook_end_col = 0, 3

        rook_row = start_row
        # Clear the rook from its original position.
        rook_start_index = get_nn_index(rook_code, rook_row, rook_start_col)
        encoding_vector[rook_start_index] = 0.0
        # Place the rook at its new position.
        rook_end_index = get_nn_index(rook_code, rook_row, rook_end_col)
        encoding_vector[rook_end_index] = 1.0

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
Function to prioritize legal moves for the search at a specific depth.
Basic move ordering using captures, killer moves and history heuristic.
This helps to find cut-off earlier for pruning, increasing efficiency.
'''
def order_moves(legal_moves, depth_from_root): 
    if not legal_moves: # check if there are any moves to order
        return legal_moves # return the empty list if no moves are available
    move_scores = {} # dictionary to store the calculated score for each move
    killer_moveIDs_at_depth = KILLER_MOVES[depth_from_root] # retrieve the two killer move IDs for the current search depth
    for move in legal_moves: # iterate over each legal move to calculate its score
        score = 0 # initialize the score for the current move
        # Capture priority: Give a high score to captures
        is_capture = move.is_captured # check if the move is a capture
        if is_capture: # if the move is a capture
            score += 100 # add a high score to prioritize captures
        # Killer move priority: give a bonus to non-capture killer moves
        if not is_capture and (move.unique_id == killer_moveIDs_at_depth[0] or move.unique_id == killer_moveIDs_at_depth[1]):
            score += 50 # add a medium bonus score to non-capture killer moves
        # History heuristic: add a score based on a move's past performance (accumulated for moves that led to a beta-cutoff)
        score += HISTORY_MOVES.get(move.unique_id, 0) # add the history score, defaulting to 0 if not found, he history table is indexed by a unique identifier for each move
        move_scores[move.unique_id] = score # Store the final calculated score for the move
    # Sort the legal moves in descending order based on their calculated score
    prioritized_moves = sorted(legal_moves, key=lambda m: move_scores.get(m.unique_id, 0), reverse=True)
    return prioritized_moves # Return the list of moves now sorted by priority

'''
Helper function to make first recursive call to best move function
'''
def get_best_move(game_state, legal_moves):
    global best_moves_list, function_calls, KILLER_MOVES, HISTORY_MOVES # Initialize the empty list of best moves and killer moves' history, next best move to None and function calls to 0
    best_moves_list = [] # Initialize the list of best moves
    function_calls = 0 # Initialize the function calls to 0
    KILLER_MOVES = defaultdict(lambda: [None, None]) # resets the killer moves table for a new search
    HISTORY_MOVES = defaultdict(int) # resets the history moves table for a new search
    root_enc = encode_board_nn(game_state.board) # encode once at root
    turn_multiplier = 1 if game_state.white_to_move else -1 # Determine if it's white's turn
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
    global best_moves_list, function_calls  # Initialize the empty best moves list, next best move to None and function calls to 0
    function_calls += 1  # Increment the function calls count
    fen_key = game_state.to_fen()  # for transposition table, creates a unique key for the board state
    alpha_orig = alpha  # transposition stores original alpha to classify the stored result

    # Check if the current position is in the transposition table
    entry = TRANSPOSITION_TABLE.get(fen_key)  # retrieves a stored entry in transposition table if it exists
    if entry and entry.get('tt_depth', -1) >= search_depth_left:  # check if the entry in transposition table is valid for the current search depth
        flag = entry.get('flag', EXACT) # gets the flags from the stored entry, defaults to EXACT if no flag is present.
        score = entry['score'] # retrieve the stored evaluation score from the entry
        if flag == EXACT: # checks if the stored score is an exact evaluation (i.e., not a bound)
            return score  # returns the exact score immediately as no further search is needed for this position
        elif flag == LOWER: # checks if the stored score is a lower bound (i.e., the best score was at least this value)
            alpha = max(alpha, score) # update the alpha value with the higher of its current value or the stored lower bound. This prunes branches that can't be better than this score.
        elif flag == UPPER: # checks if the stored score is an upper bound (i.e., the best score was at most this value)
            beta = min(beta, score) # update the beta value with the lower of its current value or the stored upper bound. This prunes branches that can't be worse than this score.
        if alpha >= beta: # checks for an alpha-beta cutoff after updating the bounds. If alpha is now greater than or equal to beta a cutoff is possible.
            return score  # return the stored score, as the current branch can be safely pruned.
    
    if not legal_moves:  # if no more legal moves
        score = turn_multiplier * terminal_eval_score(game_state)  # Evaluate the board score for the current game state and return it
        TRANSPOSITION_TABLE[fen_key] = {'score': score, 'tt_depth': search_depth_left, 'flag': EXACT}  # store a leaf node's score as EXACT
        return score

    legal_moves = order_moves(legal_moves, depth_from_root) # order legal moves at the current depth for better pruning using captures, killer moves and history scores
    # When search_depth_left == 1 i.e., batched leaf evaluation (batching per node at depth-1, not all leaves of the whole tree)
    if search_depth_left == 1:  # if we are at the second last depth of our search
        max_score = -CHECKMATE_SCORE  # set the max score to the lowest possible value
        i = 0  # i is batch start index
        while i < len(legal_moves):  # more moves remain in next batch (incase >100 legal moves exist and pruning did not occur)
            cutoff = False  # reset per batch 
            moves_batch = legal_moves[i:i + MAX_LEAF_BATCH]  # take a batch of moves to evaluate
            encoded_boards_to_evaluate = []  # list to hold encoded boards for batch evaluation
            batch_index_map = []  # list to map evaluation results back to the original move index
            move_scores_raw = [None] * len(moves_batch)  # list to store the raw scores of each move in the batch
            for j, move in enumerate(moves_batch):  # j is batch move index
                annotate_move_for_encoding(game_state, move)  # minimal annotations for nn board
                game_state.make_move(move)  # make the move on the actual game state
                next_legal_moves = game_state.get_valid_moves()  # get the legal moves from the new position
                if (len(next_legal_moves) == 0) and (game_state.is_checkmate or game_state.is_stalemate):  # If child is terminal (checkmate or stalemate) avoid NN call
                    move_scores_raw[j] = terminal_eval_score(game_state)  # directly evaluate the terminal state without a neural network call
                else:
                    # non-terminal: build the child encoding snapshot
                    new_enc = encoding.copy()  # create a copy of the parent encoding to apply changes to
                    apply_encoding(new_enc, move)  # apply the move to the encoding
                    encoded_boards_to_evaluate.append(new_enc)  # add the new encoding to the batch for evaluation
                    batch_index_map.append(j)  # record the index to map the score back
                game_state.undo_move()  # undo the move on the actual game state
            if encoded_boards_to_evaluate:  # if there are boards to evaluate
                preds = eval_batch(encoded_boards_to_evaluate)  # evaluate the batch of boards at once
                for eval_index, j in enumerate(batch_index_map):  # loop through the evaluation results
                    move_scores_raw[j] = WEIGHTS["dnn"] * float(preds[eval_index])  # store the final weighted score
            for j, move in enumerate(moves_batch):
                score = turn_multiplier * move_scores_raw[j]  # calculate the score for the current player's perspective
                # negamax check
                if score > max_score:  # If the score is greater than the current max score
                    max_score = score  # Update the maximum score
                    if search_depth_left == MAX_DEPTH:  # If the search depth is at the maximum depth
                        best_moves_list = [move]  # Initialize the list of best moves
                elif score == max_score and search_depth_left == MAX_DEPTH:  # If the score is equal to the current max score (tie)
                    best_moves_list.append(move)  # Add the move to the list of best moves
                # pruning
                if max_score > alpha:
                    alpha = max_score  # Update alpha to the maximum score found
                if alpha >= beta:  # If alpha is greater than or equal to beta
                    if not move.is_captured: # killer history updates on cutoff (non-captures only)
                        killers = KILLER_MOVES[depth_from_root] # retrieve killer moves for the current depth
                        if killers[0] != move.unique_id:
                            killers[1] = killers[0]
                            killers[0] = move.unique_id # update the killer move ID list
                        HISTORY_MOVES[move.unique_id] += search_depth_left * search_depth_left # update the history score
                    cutoff = True
                    break  # exit inner loop
            i += MAX_LEAF_BATCH  # move to the next batch of moves
            if cutoff:
                break # exit outer loop

    else:  # When search_depth_left >= 2
        max_score = -CHECKMATE_SCORE  # Initialize to a very low score for the player's best move (max score obtainable)
        for move in legal_moves:  # loop through each legal move the player can make
            new_enc = encoding.copy()  # copy encoding (fast for 768 floats)
            annotate_move_for_encoding(game_state, move)
            apply_encoding(new_enc, move)  # apply encoding changes before mutating the game state

            game_state.make_move(move)  # Make the player's move in the game state
            next_possible_moves = game_state.get_valid_moves()  # Get the valid moves for the next possible moves
            score = -negamax_alpha_beta_search(game_state, next_possible_moves, search_depth_left - 1, -beta, -alpha, -turn_multiplier, depth_from_root + 1, new_enc)  # Recursive call to find the best move for the opponent
            game_state.undo_move()  # Undo the player's move to evaluate the next one
            # negamax check
            if score > max_score:  # If the score is greater than the current max score
                max_score = score  # Update the maximum score
                if search_depth_left == MAX_DEPTH:  # If the search depth is at the maximum depth
                    best_moves_list = [move]  # Initialize the list of best moves
            elif score == max_score:  # If the score is equal to the current max score (tie)
                if search_depth_left == MAX_DEPTH:  # If the search depth is at the maximum depth
                    best_moves_list.append(move)  # Add the move to the list of best moves
            # pruning happens here
            if max_score > alpha:
                alpha = max_score  # Update alpha to the maximum score found
            if alpha >= beta:  # If alpha is greater than or equal to beta
                if not move.is_captured: # killer history updates on cutoff (non-captures only)
                    killers = KILLER_MOVES[depth_from_root] # retrieve killer moves for the current depth
                    if killers[0] != move.unique_id:
                        killers[1] = killers[0]
                        killers[0] = move.unique_id # update the killer move ID list
                    HISTORY_MOVES[move.unique_id] += search_depth_left * search_depth_left # update the history score
                break  # exit the loop (pruning)

    # classify transposition table flag based on window and then store
    if max_score <= alpha_orig:  # If the best score found is less than or equal to the original alpha, it means the search failed to raise alpha.
        flag = UPPER  # fail-low result, so the score is an upper bound on the true value (no move was found that could improve on the alpha score).
    elif max_score >= beta:  # If the best score found is greater than or equal to beta, it means the search failed to fall below beta.
        flag = LOWER  # fail-high result, so the score is a lower bound on the true value (a move is so good that the parent would have pruned this branch anyway).
    else:  # If the best score found is within the original alpha-beta window.
        flag = EXACT  # The score is considered an exact score because it's the true value for this position.
    TRANSPOSITION_TABLE[fen_key] = {'score': max_score, 'tt_depth': search_depth_left, 'flag': flag}  # store the final score with its flag
    return max_score  # Return the maximum score found

'''
Backup random AI move if AI negamax algorithm fails or returns None.
'''
def random_AI_move(legal_moves):
    return legal_moves[random.randint(0, len(legal_moves) - 1)] # Random AI move selection