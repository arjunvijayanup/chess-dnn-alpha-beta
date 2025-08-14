"""
This module provides the board encoding logic for the chess AI.
It contains all functions necessary to convert a game state into a 782-vector format for neural network inference. 
The module includes functions for creating a fresh encoding from a game state, 
efficiently updating an existing encoding after a move,
and providing helper methods for move interpretation.
This logic is designed for the prediction side, not for model training.
"""
import numpy as np

# Constants for the encoding scheme
INPUT_DIM = 782 # 768 piece planes + 1 stm + 4 castling + 8 ep-file + 1 halfmove
# A map to convert piece characters to a 0-5 integer index.
PIECE_TYPE_MAP = {'p': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5}

'''
Piece values for evaluation
These values are used to evaluate the material on the board.
'''
def square_index(row, col): # flatten (row,col) i.e. 0 to 63
     return (7 - row) * 8 + col

'''
Helper function to update the auxiliary features of the encoding vector.
This function is called by encode_board_nn
to avoid redundant code.
'''
def update_aux_features(encoding_vector, game_state):
    # set the side to move feature at index 768, +1.0 for white, -1.0 for black.
    encoding_vector[768] = 1.0 if getattr(game_state, "white_to_move", True) else -1.0
    wks, wqs, bks, bqs = get_castling_rights(game_state) # get the current castling rights for both players.
    encoding_vector[769] = 1.0 if wks else 0.0 # set the white kingside castling feature at index 769
    encoding_vector[770] = 1.0 if wqs else 0.0 # set the white queenside castling feature at index 770
    encoding_vector[771] = 1.0 if bks else 0.0 # set the black kingside castling feature at index 771
    encoding_vector[772] = 1.0 if bqs else 0.0 # set the black queenside castling feature at index 772
    for k in range(773, 781): # loop through the indices for the en-passant file (773 to 780)
        encoding_vector[k] = 0.0 # reset all en-passant file features to 0.0
    ep_file = get_en_passant_file(game_state) # get the en-passant file from the game state
    if ep_file is not None:  # if an en-passant file exists
        encoding_vector[773 + ep_file] = 1.0 # set the corresponding feature to 1.0
    half_move = get_half_move_clock(game_state) # get the half-move clock value from the game state.
    # Set the half-move clock feature at index 781
    encoding_vector[781] = min(max(half_move, 0), 100) / 100.0 # The value is clipped between 0 and 100, then scaled to a [0, 1] range

'''
One-hot encode the position into a 782-vector matching training:
768 piece planes + [stm, 4xcastling, 8xep-file, halfmove/100].
This builds a fresh encoding from scratch. Use at root, or when missing.
'''
def encode_board_nn(game_state):
    board = game_state.board
    encoded_vector = np.zeros(INPUT_DIM, dtype=np.float32)
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
    # Append auxiliary features exactly like training
    update_aux_features(encoded_vector, game_state)
    return encoded_vector

'''
Helpers to read engine state regardless of exact attribute names.
'''
# Castling-rights
def get_castling_rights(game_state):
    castling_rights = getattr(game_state, "castling_rights_current", None)
    if castling_rights is not None:
        # Get the castling rights for white kingside, white queenside, black kingside and black queenside.
        wks = getattr(castling_rights, "white_kingside", False)
        wqs = getattr(castling_rights, "white_queenside", False)
        bks = getattr(castling_rights, "black_kingside", False)
        bqs = getattr(castling_rights, "black_queenside", False)
        return bool(wks), bool(wqs), bool(bks), bool(bqs) # Return the rights as boolean values

# En-passant
def get_en_passant_file(game_state):
    ep_possible = getattr(game_state, "en_passant_possible", None)
    try:
        if isinstance(ep_possible, tuple) and len(ep_possible) == 2: # check if the en-passant information is a tuple of length 2.
            col = int(ep_possible[1]) # The column is the second element of the tuple.
            return col if 0 <= col <= 7 else None # return the column if it's a valid file (0-7)
    except Exception:
        pass
    return None # return None if the en-passant information is not valid or an exception occurred

# Half-move clock
def get_half_move_clock(game_state):
    try: # Try to convert the attribute value to an integer
        return int(getattr(game_state, "half_move_clock", 0))
    except Exception:
        return 0 # if no valid attribute is found, return 0

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
        return piece_from # piece code at the start square (e.g., 'wp' or 'bp')
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
def annotate_move_for_encoding(move):
    # Use move attributes directly from the engine's Move object
    piece_from = move.moved_piece
    captured_piece = move.captured_piece
    is_en_passant = move.is_en_passant
    is_castling = move.is_castling_move

    # Determine captured piece position for en passant since it's not the end square
    if is_en_passant: 
        captured_row, captured_col = (move.end_row + 1, move.end_col) if piece_from[0] == "w" else (move.end_row - 1, move.end_col)
    else:
        captured_row, captured_col = move.end_row, move.end_col
    
    # Handle promotion to get the correct final piece code
    piece_to = promotion_piece_code(piece_from, getattr(move, "promotion_choice", None))

    # Store these annotations as attributes on the move object for easy access later
    move._nn_pf = piece_from
    move._nn_pc = captured_piece
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