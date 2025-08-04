import random
def random_best_move(legal_moves):
    # Random AI move selection
    return legal_moves[random.randint(0, len(legal_moves) - 1)]

# Piece values and game outcome values
piece_scores = { "K" : 0,
                 "Q" : 10,
                 "R" : 5,
                 "B" : 3,
                 "N" : 3,
                 "p" : 1 }
checkmate_score = 1000
stalemate_score = 0
max_depth = 3 # set global variable for search depth

# Finding next best move based on material advantage
# This function not called anymore, we will use recursive function instead
def best_move_by_material(game_state, legal_moves):
    turn_multiplier = 1 if game_state.white_to_move else -1 # Determine if it's white's turn
    opponent_minmax_eval_score = checkmate_score # Initialize to a very high score for opponent's minimum max score (min score obtainable)
    best_player_move = None # Initialize the best player move to None
    random.shuffle(legal_moves) # Shuffle the legal moves to add randomness
    # For each legal move the player can make, evaluate the opponent's best response
    for player_move in legal_moves:
        game_state.make_move(player_move) # Make the player's move in the game state
        opponent_moves = game_state.get_valid_moves() # Get the opponent's valid moves
        opponent_max_eval_score = -checkmate_score # Initialize to a very low score for opponent's best move (max score obtainable)
        if game_state.is_stalemate: # If the game is stalemate
                eval_score = stalemate_score
        elif game_state.is_checkmate: # If the player has checkmated the opponent
                eval_score = -checkmate_score
        # Evaluate each opponent move to find the best one (Highest score)
        for opponent_move in opponent_moves: 
            game_state.make_move(opponent_move) # Make the opponent's move in the game state
            game_state.get_valid_moves() # Generate the valid moves for the opponent
            if game_state.is_checkmate: # If the opponent has checkmated the player
                eval_score = -checkmate_score
            elif game_state.is_stalemate: # If the game is stalemate
                eval_score = stalemate_score
            else: # If the game is still ongoing, evaluating score on the board based on material advantage
                eval_score = -turn_multiplier * material_eval_score(game_state.board)
            if eval_score > opponent_max_eval_score: # Finding the opponent's best move (Highest opponent score)
                opponent_max_eval_score = eval_score
            game_state.undo_move() # Undo the opponent move to evaluate the next one
        # For each player move made, if opponent's max score is less than the thier previous minimum max score, then that is the best player move
        if opponent_max_eval_score < opponent_minmax_eval_score:
            opponent_minmax_eval_score = opponent_max_eval_score # Update the opponent's minimum max score obtainable 
            best_player_move = player_move # Update this move as the best player move
        game_state.undo_move() # Undo the player's move to evaluate the next one
    return best_player_move

'''
Helper function to make first recursive call to best move function
'''
def get_best_move( game_state, legal_moves):
    global next_best_move
    next_best_move = None # Initialize the next best move to None
    # minimax_search(game_state, legal_moves, max_depth, game_state.white_to_move)
    turn_multiplier = 1 if game_state.white_to_move else -1 # Determine if it's white's turn
    negamax_search(game_state, legal_moves, max_depth, turn_multiplier) # Recursive function to find the best move
    return next_best_move # return the best move found by the minimax algorithm

'''
Minimax algorithm to find the best move (recursive function)
'''
def minimax_search(game_state, legal_moves, search_depth, white_to_move):
    global next_best_move # Initialize the next best move to None
    if search_depth == 0 or not legal_moves: # If search depth is 0 or no legal moves, return the evaluation score
        return board_eval_score(game_state.board) # Evaluate the board based on material advantage
    if white_to_move: # If it's white's turn
        max_score = -checkmate_score # Initialize to a very low score for white's best move (max score obtainable)
        for move in legal_moves: # loop through each legal move the player can make
            game_state.make_move(move) # Make the player's move in the game state
            next_possible_moves = game_state.get_valid_moves() # Get the valid moves for the next possible moves
            score = minimax_search(game_state, next_possible_moves, search_depth - 1, False) # Recursive call to find the best move for the opponent (White to move = False)
            if score > max_score:  # If the score is greater than the current max score
                max_score = score # Update the maximum score
                if search_depth == max_depth: # If the search depth is at the maximum depth
                    best_moves_list = [move] # Initialize the list of best moves
            elif score == max_score: # If the score is equal to the current max score
                if search_depth == max_depth: # If the search depth is at the maximum depth
                    best_moves_list.append(move) # Add the move to the list of best moves
            game_state.undo_move() # Undo the player's move to evaluate the next one
        if best_moves_list and search_depth == max_depth: # If there are best moves found at the maximum search depth
            next_best_move = random.choice(best_moves_list) # Randomly select one of the best moves
        return max_score # Return the maximum score found

    else:
        min_score = checkmate_score # Initialize to a very high score for
        for move in legal_moves: # loop through each legal move the opponent can make
            game_state.make_move(move) # Make the opponent's move in the game state
            next_possible_moves = game_state.get_valid_moves() # Get the valid moves for the next possible moves
            score = minimax_search(game_state, next_possible_moves, search_depth - 1, True) # Recursive call to find the best move for the player (White to move = True)
            if score < min_score: # If the score is less than the current minimum score
                min_score = score # Update the minimum score
                if search_depth == max_depth: # If the search depth is at the maximum depth
                    best_moves_list = [move] # Initialize the list of best moves
            elif score == min_score: # If the score is equal to the current minimum score
                if search_depth == max_depth: # If the search depth is at the maximum depth
                    best_moves_list.append(move) # Add the move to the list of best moves
            game_state.undo_move() # Undo the opponent's move to evaluate the next one
        if best_moves_list and search_depth == max_depth: # If there are best moves found at the maximum search depth
            next_best_move = random.choice(best_moves_list) # Randomly select one of the best moves
        return min_score # Return the minimum score found

'''
Alternate (cleaner) Negamax algorithm to find the best move (recursive function)
'''
def negamax_search(game_state, legal_moves, search_depth, turn_multiplier):
    global next_best_move # Initialize the next best move to None
    if search_depth == 0 or not legal_moves: # If search depth is 0 or no legal moves, return the evaluation score: 
        return turn_multiplier * board_eval_score(game_state)

    max_score = -checkmate_score # Initialize to a very low score for the player's best move (max score obtainable)
    for move in legal_moves: # loop through each legal move the player can make
        game_state.make_move(move) # Make the player's move in the game state
        next_possible_moves = game_state.get_valid_moves() # Get the valid moves for the next possible moves
        score = -negamax_search(game_state, next_possible_moves, search_depth - 1, -turn_multiplier) # Recursive call to find the best move for the opponent (turn_multiplier = -turn_multiplier)
        if score > max_score: # If the score is greater than the current max score
            max_score = score # Update the maximum score
            if search_depth == max_depth: # If the search depth is at the maximum depth
                best_moves_list = [move] # Initialize the list of best moves
        elif score == max_score: # If the score is equal to the current max score
            if search_depth == max_depth: # If the search depth is at the maximum depth
                best_moves_list.append(move) # Add the move to the list of best moves
        game_state.undo_move() # Undo the player's move to evaluate the next one
    if search_depth == max_depth and best_moves_list: # If there are best moves found at the maximum search depth
        next_best_move = random.choice(best_moves_list) # Randomly select one of the best moves
    return max_score # Return the maximum score found

'''
Evaluation score of the board based on material and game outcome
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
    for row in game_state.board: # Loop through each row of the chess board
        for square in row: # Loop through each square in the current row
            if square[0] == 'w': # If the piece on this square is White
                eval_score += piece_scores[square[1]] # Add the value of this White piece to the evaluation score
            elif square[0] == 'b': # If the piece on this square is Black
                eval_score -= piece_scores[square[1]] # Subtract the value of this Black piece from the evaluation score

    return eval_score # return material evaluation score

# Score of material based on material advantage
def material_eval_score(board):
    eval_score = 0 # Initialize evaluation score
    # Iterate through the board and calculate the score based on piece values
    for row in board:
        for square in row:
            if square[0] == 'w':
                eval_score += piece_scores[square[1]]
            elif square[0] == 'b':
                eval_score -= piece_scores[square[1]]
    return eval_score