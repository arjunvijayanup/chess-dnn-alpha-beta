import random

# Piece values and game outcome values
piece_scores = { "K" : 0,
                 "Q" : 10,
                 "R" : 5,
                 "B" : 3.25,
                 "N" : 3,
                 "p" : 1 }

# 2D arrays of positional scores for each piece type
# These scores are designed to encourage good piece placement and control of the board
knight_pos_scores = [ #knight needs to be placed in the center of the board to control more squares
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 2, 2, 2, 2, 2, 2, 1],
    [1, 2, 3, 3, 3, 3, 2, 1],
    [1, 3, 4, 3, 3, 4, 3, 1],
    [1, 2, 3, 4, 4, 3, 2, 1],
    [1, 2, 3, 3, 3, 3, 2, 1],
    [1, 2, 2, 2, 2, 2, 2, 1],
    [1, 1, 1, 1, 1, 1, 1, 1]
]

queen_pos_scores = [ # more open files and center control is better for queen
    [1, 1, 1, 3, 1, 1, 1, 1],
    [1, 2, 3, 3, 3, 3, 2, 1],
    [1, 4, 3, 3, 3, 3, 4, 1],
    [1, 2, 3, 3, 3, 3, 2, 1],
    [1, 2, 3, 3, 3, 3, 2, 1],
    [1, 4, 3, 3, 3, 3, 4, 1],
    [1, 2, 3, 3, 3, 3, 2, 1],
    [1, 1, 1, 3, 1, 1, 1, 1]
]

bishop_pos_scores = [ # Bishops are rewarded for central open-diagonal positions that maximize control of their own square color
    [4, 3, 2, 1, 1, 2, 3, 4],
    [3, 4, 3, 2, 2, 3, 4, 3],
    [2, 3, 4, 3, 3, 4, 3, 2],
    [1, 2, 3, 4, 4, 3, 2, 1],
    [1, 2, 3, 4, 4, 3, 2, 1],
    [2, 3, 4, 3, 3, 4, 3, 2],
    [3, 4, 3, 2, 2, 3, 4, 3],
    [4, 3, 2, 1, 1, 2, 3, 4]
]

rook_pos_scores = [ # Rooks get higher scores for open files and central squares helping them move and work together better
    [4, 3, 4, 4, 4, 4, 3, 4],
    [4, 4, 4, 4, 4, 4, 4, 4],
    [1, 1, 2, 3, 3, 2, 1, 1],
    [1, 2, 3, 4, 4, 3, 2, 1],
    [1, 2, 3, 4, 4, 3, 2, 1],
    [1, 1, 2, 3, 3, 2, 1, 1],
    [4, 4, 4, 4, 4, 4, 4, 4],
    [4, 3, 4, 4, 4, 4, 3, 4]
]

white_pawn_pos_scores = [ # Reward passed pawns and pawns in the center of the board from white's perspective
    [8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 8],
    [5, 6, 6, 7, 7, 6, 6, 5],
    [2, 3, 3, 5, 5, 3, 3, 2],
    [1, 2, 3, 4, 4, 3, 2, 1],
    [1, 1, 2, 3, 3, 2, 1, 1],
    [1, 1, 1, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0]
]

black_pawn_pos_scores = [ # Reward passed pawns and pawns in the center of the board from black's perspective
    [0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 1, 1, 1],
    [1, 1, 2, 3, 3, 2, 1, 1],
    [1, 2, 3, 4, 4, 3, 2, 1],
    [2, 3, 3, 5, 5, 3, 3, 2],
    [5, 6, 6, 7, 7, 6, 6, 5],
    [8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 8]
]

position_scores = {
    "N": knight_pos_scores,
    "Q": queen_pos_scores,
    "B": bishop_pos_scores,
    "R": rook_pos_scores,
    "bp": white_pawn_pos_scores,
    "wp": black_pawn_pos_scores
}

checkmate_score = 1000
stalemate_score = 0
max_depth = 3 # set global variable for search depth


'''
Evaluation score of the board based on material, game outcome and positional advantage
Here positive score is good for white and a negative score is good for black
'''
def board_eval_score(game_state):
    if game_state.is_checkmate: # If the game is checkmate
        if game_state.white_to_move: # If it's white's turn and the game is checkmate
            return -checkmate_score  # black wins
        else: # If it's black's turn and the game is checkmate
            return checkmate_score   # white wins
    elif game_state.is_stalemate: # If the game is in stalemate
        return stalemate_score
    
    eval_score = 0  # Initialize material evaluation score
    # Loop through every square on the board
    for row in range(len(game_state.board)): # 
        for col in range(len(game_state.board[row])):
            square = game_state.board[row][col] # get the piece on the square
            if square != "--": # Only evaluate if piece present
                position_score = 0 # initialize positional bonus/penalty
                if square[1] != "K":  # Don't give positional scores to the king
                    if square[1] == "p":  # If it's a pawn use pawn-specific table (for its color)
                        position_score = position_scores[square][row][col] # get position score based on piece
                    else: # other pieces
                        position_score = position_scores[square[1]][row][col]
                # Add the piece value and positional score to the evaluation score
                # If it's white's turn, add the piece value and positional score, else subtract it
                if square[0] == 'w':
                    eval_score += piece_scores[square[1]] + position_score * 0.1
                elif square[0] == 'b':
                    eval_score -= piece_scores[square[1]] + position_score * 0.1
    return eval_score

'''
Helper function to make first recursive call to best move function
'''
def get_best_move( game_state, legal_moves, return_queue):
    global next_best_move, function_calls # Initialize the next best move to None and function calls to 0
    next_best_move = None # Initialize the next best move to None
    function_calls = 0 # Initialize the function calls to 0
    turn_multiplier = 1 if game_state.white_to_move else -1 # Determine if it's white's turn
    random.shuffle(legal_moves) # Shuffle the legal moves to add randomness
    negamax_alpha_beta_search(game_state, legal_moves, max_depth, -checkmate_score, checkmate_score, turn_multiplier) # Recursive function to find the best move (pass maximum alpha beta)
    # print(f"Number of function calls made: {function_calls}") # check performance of the algorithm
    return return_queue.put(next_best_move) # Return the next best move found by the algorithm


'''
Negamax algorithm to find the best move (recursive function) with alpha-beta pruning (improved efficiency)
'''

def negamax_alpha_beta_search(game_state, legal_moves, search_depth, alpha, beta, turn_multiplier):
    global next_best_move, function_calls # Initialize the next best move to None and function calls to 0
    function_calls += 1 # Increment the function calls count
    best_moves_list = [] # Initialize the list of best moves
    if search_depth == 0 or not legal_moves: # If search depth is 0 or no legal moves, return the evaluation score: 
        return turn_multiplier * board_eval_score(game_state)

    max_score = -checkmate_score # Initialize to a very low score for the player's best move (max score obtainable)
    for move in legal_moves: # loop through each legal move the player can make
        game_state.make_move(move) # Make the player's move in the game state
        next_possible_moves = game_state.get_valid_moves() # Get the valid moves for the next possible moves
        score = -negamax_alpha_beta_search(game_state, next_possible_moves, search_depth - 1, -beta, -alpha, -turn_multiplier) # Recursive call to find the best move for the opponent
        if score > max_score: # If the score is greater than the current max score
            max_score = score # Update the maximum score
            if search_depth == max_depth: # If the search depth is at the maximum depth
                best_moves_list = [move] # Initialize the list of best moves
        elif score == max_score: # If the score is equal to the current max score
            if search_depth == max_depth: # If the search depth is at the maximum depth
                best_moves_list.append(move) # Add the move to the list of best moves
        game_state.undo_move() # Undo the player's move to evaluate the next one
        if max_score > alpha:  # pruning happens here
            alpha = max_score # Update alpha to the maximum score found
        if alpha >= beta: # If alpha is greater than or equal to beta
            break # exit the loop (pruning)
    if search_depth == max_depth and best_moves_list: # If there are best moves found at the maximum search depth
        next_best_move = random.choice(best_moves_list) # Randomly select one of the best moves
    return max_score # Return the maximum score found


''' 
Backup random AI move if AI negamax algorithm fails or is not implemented.
'''
def random_AI_move(legal_moves):
    # Random AI move selection
    return legal_moves[random.randint(0, len(legal_moves) - 1)]