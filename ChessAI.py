import random
def random_AI_move(legal_moves):
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

# Finding next best AI move based on material advantage
def best_AI_move(game_state, legal_moves):
    turn_multiplier = 1 if game_state.white_to_move else -1 # Determine if it's white's turn
    opponent_minmax_eval_score = checkmate_score # Initialize to a very high score for opponent's minimum max score (min score obtainable)
    best_player_move = None # Initialize the best player move to None
    random.shuffle(legal_moves) # Shuffle the legal moves to add randomness
    # For each legal move the player can make, evaluate the opponent's best response
    for player_move in legal_moves:
        game_state.make_move(player_move) # Make the player's move in the game state
        opponent_moves = game_state.get_valid_moves() # Get the opponent's valid moves
        opponent_max_eval_score = -checkmate_score # Initialize to a very low score for opponent's best move (max score obtainable)
        # Evaluate each opponent move to find the best one (Highest score)
        for opponent_move in opponent_moves: 
            game_state.make_move(opponent_move) # Make the opponent's move in the game state
            if game_state.is_checkmate: # If the opponent has checkmated the player
                eval_score = -turn_multiplier * checkmate_score #
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