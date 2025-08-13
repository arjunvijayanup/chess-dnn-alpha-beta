"""
This file contains the chess AI's core logic, including the Negamax search algorithm
with alpha-beta pruning and a neural network-based evaluation function.
"""
import numpy as np # for numeric array operation (encoding)
import tensorflow as tf # for loading and running the neural network model
import random # for random move selection in fallback logic
from collections import defaultdict # for using killer moves' history tables
import os  # for model file existence check
import chessEncoding

# Initialize the opening book for the AI
from chessOpening import OpeningBook
BOOK = OpeningBook(
    hf_name="Lichess/chess-openings",
    split="train",
    max_book_plies=20,
    temperature=0.75,
    cache_file="opening_prefix.pkl.gz",  # optional cache for speed
    force_rebuild=False,
    limit=None,
    random_seed=None,
    verbose=False
) # Create an instance of the OpeningBook class

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
MODEL_INPUT = None # Preallocated input buffer for the model (1 × 782)
prediction_graph = None # Compiled prediction graph for speed (if available)

# transposition table
TRANSPOSITION_TABLE = {} # global dictionary for storing board states and their evaluation scores
EXACT, LOWER, UPPER = 0, 1, 2 # transposition flags to classify the stored score as exact, a lower bound, or an upper bound

# Killer moves for each ply (search depth)
# History of good moves (large table for long-term storage)
KILLER_MOVES = defaultdict(lambda: [None, None]) # two killers per ply (auto-creates [None, None] for unseen ply), stores non-capture moves that caused a beta cutoff at a given depth
HISTORY_MOVES = defaultdict(int) # MOVES: stores a score for each move, incremented when the move causes a cutoff

# Ping-pong penalty and score thresholds for a "drawish" position
PING_PONG_PENALTY = 0.02  # penalty to discourage back and forth loops(NN interval: [-1,1])
DRAWISH_EPSILON   = 0.15  # only discourage ping-pong if position is roughly equal

'''
Vectorized DNN eval evaluation
This uses a 782-vector encoding (768 piece planes + 14 aux features).
'''
def eval_batch(encoded_boards):
    input_array = np.asarray(encoded_boards, dtype=np.float32) # Convert to numpy array
    if input_array.ndim == 1: # if a single board is provided, reshape to 2D
        input_array = input_array[np.newaxis, :] # reshape to 1x782
    if prediction_graph is not None: # use compiled graph for speed
        predictions = prediction_graph(input_array) # run the model on the input array
    else: # fallback to eager execution
        predictions = eval_model(input_array, training=False) # run the model on the input array
    predictions = np.asarray(predictions) # Convert predictions to numpy array
    if predictions.ndim == 2: # if predictions are 2D, take the first column (single output per board)
        predictions = predictions[:,0] # take the first column
    return predictions # return the predictions as a 1D array

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
    if not os.path.exists(MODEL_PATH):
        print(f"[chessAI] Model file not found: '{MODEL_PATH}'.", flush=True)
    eval_model = tf.keras.models.load_model(MODEL_PATH, compile=False) # Load pre-trained Keras model from disk (compiled=False skips training setup)
    MODEL_INPUT = np.zeros((1, chessEncoding.INPUT_DIM), dtype=np.float32) # Preallocate buffer for model input (1 × 782)
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
        is_capture = (getattr(move, 'captured_piece', '--') != '--') # check if the move is a capture
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

"""
This function checks if the game state is in an immediate two-fold repetition,
following an ABAB pattern over the last four moves.
It checks for two types of repetitive moves and applies a penalty only when the current
position is considered a drawish spot.
"""
def ping_pong_penalty_needed(game_state, move, score, proposed_key=None):
# Gate the penalty strictly to drawish spots: never discourage repetition when worse
    if abs(score) > DRAWISH_EPSILON:
        return False

    # Check for an immediate two-fold position cycle (A>B>A) using the position history.
    position_history = getattr(game_state, "position_stack", None)
    # The cycle requires at least four positions in the history.
    # Immediate bounce back to the position from two plies ago (AB) -> A
    if position_history is not None and len(position_history) >= 2 and proposed_key is not None:
        if proposed_key == position_history[-2]:
            return True

    # Check for a same-side, same-piece immediate bounce (A->B then B->A).
    # This is a different type of repetition that is not position-based.
    move_log = getattr(game_state, "moves_log", None)
    # A bounce requires at least two moves by the same player.
    if not move_log or len(move_log) < 2:
        return False
        
    # Get the previous move made by the current player.
    last_move_by_same_player = move_log[-2]

    # Check if the same piece (color and type) is making the move.
    try:
        if last_move_by_same_player.moved_piece[0] != move.moved_piece[0]:
            return False  # Colors don't match.
        if last_move_by_same_player.moved_piece[1] != move.moved_piece[1]:
            return False  # Piece types don't match.
    except Exception:
        return False  # Handle cases where piece data is missing.

    # Check if the move is an exact reversal of the previous move.
    if not (last_move_by_same_player.start_row == move.end_row and
            last_move_by_same_player.start_col == move.end_col and
            last_move_by_same_player.end_row == move.start_row and
            last_move_by_same_player.end_col == move.start_col):
        return False

    # Ensure no captures, castling, or pawn promotions occurred on either move.
    if getattr(last_move_by_same_player, 'is_captured', False) or getattr(move, 'is_captured', False):
        return False
    if getattr(last_move_by_same_player, 'is_castling_move', False) or getattr(move, 'is_castling_move', False):
        return False
    if getattr(last_move_by_same_player, 'is_pawn_promotion', False) or getattr(move, 'is_pawn_promotion', False):
        return False

    # If all checks pass, a ping-pong penalty is needed.
    return True

'''
Helper function to make first recursive call to best move function
'''
def get_best_move(game_state, legal_moves):
    # Opening book early return
    try: # Attempt to use the opening book
        if BOOK.enabled: # Check if the opening book is enabled
            pre, cand = BOOK.peek(game_state) # Get the current move prefix and candidate moves from the book
            #if len(pre) <= 2: # Only print verbose information for early plies
            #    print("[Book] prefix:", pre) # Print the current move prefix
            #    print("[Book] raw candidates:", list(cand.items())[:10]) # Print the top 10 raw candidates from the book

            bm = BOOK.pick(game_state, legal_moves) # Try to pick a move from the opening book that is also legal

            if bm is not None: # If a book move is found
                return bm # Return the chosen book move
    except Exception as e: # Catch any exceptions that occur during book lookup
        print("[Book] error:", repr(e)) # Print the error

    global best_moves_list, function_calls, KILLER_MOVES, HISTORY_MOVES # Initialize the empty list of best moves and killer moves' history, next best move to None and function calls to 0
    TRANSPOSITION_TABLE.clear()
    best_moves_list = [] # Initialize the list of best moves
    function_calls = 0 # Initialize the function calls to 0
    KILLER_MOVES = defaultdict(lambda: [None, None]) # resets the killer moves table for a new search
    HISTORY_MOVES = defaultdict(int) # resets the history moves table for a new search
    root_enc = chessEncoding.encode_board_nn(game_state) # encode once at root (includes side-to-move/stm and castling/ep/halfmove)
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
    beta_orig = beta
    penalized_any = False  # track if any child score at this node was ping-pong penalized

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
    
    # Early draw detection: treat repetition/50-move/insufficient material as terminal draws
    if (getattr(game_state, 'is_threefold_repetition', False) or 
        getattr(game_state, 'is_fifty_move_draw', False) or
        getattr(game_state, 'is_insufficient_material', False)):
        score = turn_multiplier * STALEMATE_SCORE
        TRANSPOSITION_TABLE[fen_key] = {'score': score, 'tt_depth': search_depth_left, 'flag': EXACT}
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
            proposed_keys = [None] * len(moves_batch)  # repetition keys after making each candidate move
            batch_index_map = []  # list to map evaluation results back to the original move index
            move_scores_raw = [None] * len(moves_batch)  # list to store the raw scores of each move in the batch
            for j, move in enumerate(moves_batch):  # j is batch move index
                chessEncoding.annotate_move_for_encoding(move)  # minimal annotations for nn board
                game_state.make_move(move)  # make the move on the actual game state
                next_legal_moves = game_state.get_valid_moves()  # get the legal moves from the new position
                if getattr(game_state, "position_stack", None): # record the repetition key for THIS candidate position (used to test ABAB continuation)
                    proposed_keys[j] = game_state.position_stack[-1]
                if (len(next_legal_moves) == 0) and (game_state.is_checkmate or game_state.is_stalemate):  # If child is terminal (checkmate or stalemate) avoid NN call
                    move_scores_raw[j] = terminal_eval_score(game_state)  # directly evaluate the terminal state without a neural network call
                elif (getattr(game_state, 'is_threefold_repetition', False) or
                      getattr(game_state, 'is_fifty_move_draw', False) or
                      getattr(game_state, 'is_insufficient_material', False)):
                    move_scores_raw[j] = STALEMATE_SCORE
                else:
                    # non-terminal: build the child encoding snapshot
                    new_enc = encoding.copy()  # create a copy of the parent encoding to apply changes to
                    chessEncoding.apply_encoding(new_enc, move)  # apply the move to the encoding
                    chessEncoding.update_aux_features(new_enc, game_state)  # sync stm/castling/ep/halfmove after move
                    encoded_boards_to_evaluate.append(new_enc)  # add the new encoding to the batch for evaluation
                    batch_index_map.append(j)  # record the index to map the score back
                game_state.undo_move()  # undo the move on the actual game state
            if encoded_boards_to_evaluate:  # if there are boards to evaluate
                preds = eval_batch(encoded_boards_to_evaluate)  # evaluate the batch of boards at once
                for eval_index, j in enumerate(batch_index_map):  # loop through the evaluation results
                    raw = float(preds[eval_index]) # White-Centric NN output
                    move_scores_raw[j] = WEIGHTS["dnn"] * raw
            for j, move in enumerate(moves_batch):
                score = turn_multiplier * move_scores_raw[j]  # calculate the score for the current player's perspective
                if ping_pong_penalty_needed(game_state, move, score, proposed_key=proposed_keys[j]): # cycle OR same-piece bounce check and only in drawish spots
                    score -= PING_PONG_PENALTY
                    penalized_any = True
                # negamax check
                if score > max_score:  # If the score is greater than the current max score
                    max_score = score  # Update the maximum score
                # pruning
                if max_score > alpha:
                    alpha = max_score  # Update alpha to the maximum score found
                if alpha >= beta:  # If alpha is greater than or equal to beta
                    if getattr(move, 'captured_piece', '--') == '--': # killer history updates on cutoff (non-captures only)
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
            new_enc = encoding.copy()  # copy encoding (fast for 782 floats)
            chessEncoding.annotate_move_for_encoding(move)
            chessEncoding.apply_encoding(new_enc, move)  # apply encoding changes before mutating the game state

            game_state.make_move(move)  # Make the player's move in the game state
            chessEncoding.update_aux_features(new_enc, game_state)  # sync aux features for the child
            next_possible_moves = game_state.get_valid_moves()  # Get the valid moves for the next possible moves
            score = -negamax_alpha_beta_search(game_state, next_possible_moves, search_depth_left - 1, -beta, -alpha, -turn_multiplier, depth_from_root + 1, new_enc)  # Recursive call to find the best move for the opponent
            game_state.undo_move()  # Undo the player's move to evaluate the next one
            # negamax check
            if score > max_score:  # If the score is greater than the current max score
                max_score = score  # Update the maximum score
                if depth_from_root == 0:  # If the search depth is at the maximum depth
                    best_moves_list = [move]  # Initialize the list of best moves
            elif score == max_score:  # If the score is equal to the current max score (tie)
                if depth_from_root == 0:  # If the search depth is at the maximum depth
                    best_moves_list.append(move)  # Add the move to the list of best moves
            # pruning happens here
            if max_score > alpha:
                alpha = max_score  # Update alpha to the maximum score found
            if alpha >= beta:  # If alpha is greater than or equal to beta
                if getattr(move, 'captured_piece', '--') == '--': # killer history updates on cutoff (non-captures only)
                    killers = KILLER_MOVES[depth_from_root] # retrieve killer moves for the current depth
                    if killers[0] != move.unique_id:
                        killers[1] = killers[0]
                        killers[0] = move.unique_id # update the killer move ID list
                    HISTORY_MOVES[move.unique_id] += search_depth_left * search_depth_left # update the history score
                break  # exit the loop (pruning)

    # classify transposition table flag based on window and then store
    if max_score <= alpha_orig:  # If the best score found is less than or equal to the original alpha, it means the search failed to raise alpha.
        flag = UPPER  # fail-low result, so the score is an upper bound on the true value (no move was found that could improve on the alpha score).
    elif max_score >= beta_orig:  # If the best score found is greater than or equal to beta, it means the search failed to fall below beta.
        flag = LOWER  # fail-high result, so the score is a lower bound on the true value (a move is so good that the parent would have pruned this branch anyway).
    else:  # If the best score found is within the original alpha-beta window.
        flag = EXACT  # The score is considered an exact score because it's the true value for this position.
    if penalized_any and flag == EXACT:
        flag = LOWER  # avoid caching history-tainted EXACT scores
    TRANSPOSITION_TABLE[fen_key] = {'score': max_score, 'tt_depth': search_depth_left, 'flag': flag}  # store the final score with its flag
    return max_score  # Return the maximum score found

'''
Backup random AI move if AI negamax algorithm fails or returns None.
'''
def random_AI_move(legal_moves):
    return legal_moves[random.randint(0, len(legal_moves) - 1)] # Random AI move selection