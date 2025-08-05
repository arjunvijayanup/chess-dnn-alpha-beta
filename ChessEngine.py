"""
Stores the GameState class, which is storing the current state of the chess game. Also,
helps the engine to determine the legal moves for each piece and the current state of the game.
"""
class GameState():

    def __init__(self):
        # 8x8 board, with each piece represented by a two-character string.
        # The first character is the color ('b' for black, 'w' for white).
        # The second character is the piece type ('R' for rook, 'N' for knight, 'B' for bishop,
        # 'Q' for queen, 'K' for king, 'p' for pawn).
        # '--' represents an empty square.
        self.board = [
            ["bR", "bN", "bB", "bQ", "bK", "bB", "bN", "bR"],
            ["bp", "bp", "bp", "bp", "bp", "bp", "bp", "bp"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["wp", "wp", "wp", "wp", "wp", "wp", "wp", "wp"],
            ["wR", "wN", "wB", "wQ", "wK", "wB", "wN", "wR"],
        ]
        # Initializing move dictionary to map piece types to their respective move functions
        self.move_function_map = {'p': self.get_pawn_moves, 'R': self.get_rook_moves,
                                  'N': self.get_knight_moves, 'B': self.get_bishop_moves,
                                  'Q': self.get_queen_moves, 'K': self.get_king_moves}
        self.white_to_move = True # True if it's white's turn, False if it's black's turn
        self.moves_log = [] # List keeping track of movements taking place
        self.white_king_pos = (7, 4) # White king location
        self.black_king_pos = (0, 4) # Black king location
        self.is_in_check = False    # If current player is in check, then True
        self.pinned_pieces = []  # Pieces that are pinned (To ensure pinned pieces are not moved)
        self.checking_pieces = []   # Pieces that are in check
        self.is_checkmate = False
        self.is_stalemate = False
        self.en_passant_possible = () # square where en passant capture happens
        self.en_passant_log = [self.en_passant_possible]
        # Castling rights
        self.castling_rights_current = CastleRights(True, True, True, True)
        self.castling_rights_log = [CastleRights(self.castling_rights_current.white_kingside, self.castling_rights_current.black_kingside,
                                                 self.castling_rights_current.white_queenside, self.castling_rights_current.black_queenside)] # copy castling rights from its original state and keep track of the changes

    '''
    Takes moves as a parameter and executes it.
    This will not work for castling, enpassant or pawn promotion.
    '''
    def make_move(self, move, promotion_choice='Q'):
        # Update the board and game state by making the given move
        self.board[move.start_row][move.start_col] = "--"   # setting start location as blank after the move done
        self.board[move.end_row][move.end_col] = move.moved_piece  # Move the selected piece to the end square
        self.moves_log.append(move)   # Log the move (Used to undo later)
        self.white_to_move = not self.white_to_move   # Swapping players
        # Updating king's location if moved
        if move.moved_piece == 'wK':
            self.white_king_pos = (move.end_row, move.end_col)
        elif move.moved_piece == 'bK':
            self.black_king_pos = (move.end_row, move.end_col)
        # If pawn moves two places, next move can capture en-passant
        if move.moved_piece[1] == 'p' and abs(move.start_row - move.end_row) == 2:
            self.en_passant_possible = ((move.end_row + move.start_row)//2, move.start_col)
        else:
            self.en_passant_possible = () # resetting en-passant square if pawn did not move two squares
        # If en-passant move done, must update the board to capture the pawn
        if move.is_en_passant:
            self.board[move.start_row][move.end_col] = "--"
        
        # if pawn promotion changed piece
        if move.is_pawn_promotion:
            self.board[move.end_row][move.end_col] = move.moved_piece[0] + promotion_choice

        # Castling move
        if move.is_castling_move:
            if move.end_col - move.start_col == 2:    # Kingside castling
                self.board[move.end_row][move.end_col - 1] = self.board[move.end_row][move.end_col + 1]   # moves rook after castling
                self.board[move.end_row][move.end_col + 1] = '--' # Erase the old rook
            else:   # Queenside castling
                self.board[move.end_row][move.end_col + 1] = self.board[move.end_row][move.end_col - 2]   # moves rook after castling
                self.board[move.end_row][move.end_col - 2] = '--' # Erase the old rook

        self.en_passant_log.append(self.en_passant_possible) # Keeping track of the en-passant square changes

        # Updating castling rights - whenever rook or king moves
        self.update_castling_rights(move)
        self.castling_rights_log.append(CastleRights(self.castling_rights_current.white_kingside, self.castling_rights_current.black_kingside,
                                                     self.castling_rights_current.white_queenside, self.castling_rights_current.black_queenside)) # copy castling rights from its original state and keep track of the changes

    def undo_move(self):
        # Undo the last move made
        if len(self.moves_log)!=0:    # making sure that move log is not empty
            move = self.moves_log.pop()   # remove last move log
            self.board[move.start_row][move.start_col] = move.moved_piece  # Resetting the moved piece to its original square
            self.board[move.end_row][move.end_col] = move.captured_piece   # Retrieving the captured piece back to its original square
            self.white_to_move = not self.white_to_move   # Swapping players back to the previous player

            # Restoring king's location (white or black) if moved
            if move.moved_piece == 'wK':
                self.white_king_pos = (move.start_row, move.start_col)
            elif move.moved_piece == 'bK':
                self.black_king_pos = (move.start_row, move.start_col)
            # Undo en-passant
            if move.is_en_passant:
                self.board[move.end_row][move.end_col] = "--" # remvoing pawn added in the wrong square
                self.board[move.start_row][move.end_col] = move.captured_piece   # puts the pawn back into the square it was captured from
            # Undo en-passant rights
            self.en_passant_log.pop() # Getting rid of the new en-passant square from the move we are undoing
            self.en_passant_possible = self.en_passant_log[-1] # Setting current en-passant square to the last one in the list
            # Undo castling rights
            self.castling_rights_log.pop()  # Getting rid of the new castle rihts from the move we are undoing
            self.castling_rights_current =  self.castling_rights_log[-1] # Setting current castling rights to the last one in the list
            # Undo castle move
            if move.is_castling_move:
                if move.end_col - move.start_col == 2:   # Kingside castling
                    self.board[move.end_row][move.end_col + 1] = self.board[move.end_row][move.end_col - 1]   # moves rook back to original position
                    self.board[move.end_row][move.end_col - 1] = '--' # Erase the earlier castled rook square
                else:   # Queenside castling
                    self.board[move.end_row][move.end_col - 2] = self.board[move.end_row][move.end_col + 1]   # moves rook back to original position
                    self.board[move.end_row][move.end_col + 1] = '--' # Erase the earlier castled rook square

            self.is_checkmate = False # when we undo move, we are not in checkmate anymore
            self.is_stalemate = False # when we undo move, we are not in stalemate anymore

    def update_castling_rights(self, move):
        # Update the castle rights given the move
        if move.captured_piece == "wR":
            if move.end_col == 0:  # left rook
                self.castling_rights_current.white_queenside = False
            elif move.end_col == 7:  # right rook
                self.castling_rights_current.white_kingside = False
        elif move.captured_piece == "bR":
            if move.end_col == 0:  # left rook
                self.castling_rights_current.black_queenside = False
            elif move.end_col == 7:  # right rook
                self.castling_rights_current.black_kingside = False

        if move.moved_piece == 'wK':
            self.castling_rights_current.white_kingside = False
            self.castling_rights_current.white_queenside = False
        elif move.moved_piece == 'bK':
            self.castling_rights_current.black_kingside = False
            self.castling_rights_current.black_queenside = False
        elif move.moved_piece == 'wR':
            if move.start_row == 7:
                if move.start_col == 0:  # this is the left rook
                    self.castling_rights_current.white_queenside = False
                elif move.start_col == 7:   # right rook
                    self.castling_rights_current.white_kingside = False
        elif move.moved_piece == 'bR':
            if move.start_row == 0:
                if move.start_col == 0:  # this is the left rook
                    self.castling_rights_current.black_queenside = False
                elif move.start_col == 7:   # right rook
                    self.castling_rights_current.black_kingside = False

    def get_valid_moves(self):
        temp_castling_rights = CastleRights(self.castling_rights_current.white_kingside, self.castling_rights_current.black_kingside,
                                            self.castling_rights_current.white_queenside, self.castling_rights_current.black_queenside)   # Copying current castling rights
        move_list = []
        self.is_in_check, self.pinned_pieces, self.checking_pieces = self.check_for_pins_and_checks()

        if self.white_to_move:
            king_row = self.white_king_pos[0]
            king_col = self.white_king_pos[1]
        else:
            king_row = self.black_king_pos[0]
            king_col = self.black_king_pos[1]
        if self.is_in_check:
            if len(self.checking_pieces) == 1: # In case of One check, block check or move king
                move_list = self.get_all_possible_moves()
                # For blocking a check, a piece must be moved into one of the squares b/w enemy piece and king
                check_info = self.checking_pieces[0] # checking info
                checking_piece_row = check_info[0]
                checking_piece_col = check_info[1]
                checking_piece = self.board[checking_piece_row][checking_piece_col] # Enemy piece which caused the check
                valid_squares = [] # squares that the pieces that can move towards
                # If knight, it must capture the knight or move the king, or other pieces can be used to block
                if checking_piece[1] == 'N':
                    valid_squares = [(checking_piece_row, checking_piece_col)]
                else:
                    for step in range(1, 8):
                        # if check[2] and check[3] are check directions
                        valid_square = (king_row + check_info[2] * step, king_col + check_info[3] * step)
                        valid_squares.append(valid_square) # All positions between king and checking enemy piece are added
                        # once the piece position is reached and checked
                        if valid_square[0] == checking_piece_row and valid_square[1] == checking_piece_col:
                            break
                # Getting rid of any moves that dont block the check or king movement
                for move_index in range(len(move_list)-1, -1, -1): # Going in backwards search direction when removing from list
                    if move_list[move_index].moved_piece[1] != 'K': # Move does not move king so it must block or capture
                        if not (move_list[move_index].end_row, move_list[move_index].end_col) in valid_squares: # if move does not block check or capture piece
                            move_list.remove(move_list[move_index])
            else: # Double check, in this case king has to move
                self.get_king_moves(king_row, king_col, move_list)
        else: # No check, so all moves are fine
            move_list = self.get_all_possible_moves()
            if self.white_to_move:
                self.get_castle_moves(self.white_king_pos[0], self.white_king_pos[1], move_list)
            else:
                self.get_castle_moves(self.black_king_pos[0], self.black_king_pos[1], move_list)

        if len(move_list) == 0:
            if self.is_in_check:
                self.is_checkmate = True
            else:
                self.is_stalemate = True
        else:
            self.is_checkmate = False
            self.is_stalemate = False

        self.castling_rights_current = temp_castling_rights
        return move_list

    def is_king_in_check(self):
        # Check if the current player's king is in check
        if self.white_to_move:
            # Check if the white king is in check
            return self.is_square_under_attack(self.white_king_pos[0], self.white_king_pos[1])
        else:
            # Check if the black king is in check
            return self.is_square_under_attack(self.black_king_pos[0], self.black_king_pos[1])

    def is_square_under_attack(self, row, col):
        # Determine if the enemy can attack the square at (row, col))
        self.white_to_move = not self.white_to_move  # Switch to the opponent's turn
        opponent_moves = self.get_all_possible_moves()  # Get all possible moves for the opponent
        self.white_to_move = not self.white_to_move  # Switch back to the current player's turn
        for move in opponent_moves:  # Check if any of the opponent's moves can attack the square
            if move.end_row == row and move.end_col == col:  # If the opponent can attack the square
                return True  # The square is under attack
        return False  # The square is not under attack

    def get_all_possible_moves(self):
        # All moves that can be made by the current player without considering checks
        move_list = []  # List to store all possible moves
        for row in range(len(self.board)): # Loop through each row
            for col in range(len(self.board[row])):  # Loop through each column in the row
                piece_color = self.board[row][col][0]  # Get the color of the piece at (row, col))
                if (piece_color =='w' and self.white_to_move) or (piece_color == 'b' and not self.white_to_move):  # Check if it's the player's turn
                    piece_char = self.board[row][col][1]   # Get the piece type at (row, col))
                    self.move_function_map[piece_char](row, col, move_list)  # Call the appropriate move function for the piece
        return move_list  # Return the list of all possible moves

    def check_for_pins_and_checks(self):
        found_pins = []  # List to keep track of pinned pieces
        found_checks = []  # Squares where enemy is applying check from
        is_currently_in_check = False
        if self.white_to_move:
            opponent_color = 'b'
            player_color = 'w'
            king_row = self.white_king_pos[0]
            king_col = self.white_king_pos[1]
        else:
            opponent_color = 'w'
            player_color = 'b'
            king_row = self.black_king_pos[0]
            king_col = self.black_king_pos[1]
        # Checking all directions surrounding the kings for pins and checks, and also keep track of the pins
        directions = ((-1, 0), (0, -1), (1, 0), (0, 1),  # orthognals: up, left, down, right
                      (-1, -1), (-1, 1), (1, -1), (1, 1))  # diagonals: up-left, up-right, down-left, down-right
        for direction_index in range (len(directions)):
            direction = directions[direction_index]
            potential_pin = () # resetting possible pins
            for step in range (1,8):
                target_row = king_row + direction[0] * step
                target_col = king_col + direction[1] * step
                if 0 <= target_row < 8 and 0 <= target_col < 8:
                    target_piece = self.board[target_row][target_col]
                    if target_piece[0] == player_color and target_piece[1] != 'K': # endPiece[1] != K added to avoid real king protecting phantom king
                        if potential_pin == (): # First allied piece could be pinned
                            potential_pin = (target_row, target_col, direction[0], direction[1])
                        else: # Second allied piece, so no pin needed or check in this direction
                            break
                    elif target_piece[0] == opponent_color:
                        piece_type = target_piece[1]
                        # Possibilities in this condition are as follows:
                        # 1. Orthogonal from king and the piece is a Rook
                        # 2. Diagonal from the king and the piece is a Bishop
                        # 3. 1 Square away diagonal from king and piece is pawn
                        # 4. Any direction from king and piece is Queen
                        # 5. Any direction and 1 square away and piece is king
                        #    Necessary to prevent a king move to a square which is controlled by another king
                        if (0 <= direction_index <= 3 and piece_type == 'R') or \
                           (4 <= direction_index <= 7 and piece_type == 'B') or \
                           (step == 1 and piece_type == 'p' and ((opponent_color == 'w' and 6 <= direction_index <= 7)or (opponent_color == 'b' and 4 <= direction_index <= 5))) or \
                           (piece_type == 'Q') or (step == 1 and piece_type == 'K'):
                            if potential_pin == (): # no piece blocking, so checking
                                is_currently_in_check = True
                                found_checks.append((target_row, target_col, direction[0], direction[1]))
                                break
                            else: # if piece blocking, pin
                                found_pins.append(potential_pin)
                                break
                        else: # if enemy piece not doing check
                            break
                else: # out of bounds of chess board
                    break
        # Checking for knight checks
        knight_offsets = ((-2, -1), (-2, 1), (-1, -2), (-1, 2),
                          (1, -2), (1, 2), (2, -1), (2, 1))
        for move_offset in knight_offsets:
            target_row = king_row + move_offset[0]
            target_col = king_col + move_offset[1]
            if 0 <= target_row < 8 and 0 <= target_col < 8:
                target_piece = self.board[target_row][target_col]
                if target_piece[0] == opponent_color and target_piece[1] == 'N': # enemy knoight attacking King
                    is_currently_in_check = True
                    found_checks.append((target_row, target_col, move_offset[0], move_offset[1]))

        return is_currently_in_check, found_pins, found_checks

    def get_pawn_moves(self, row, col, move_list):
        # Getting pawn moves and appending them to the moves list
        piecePinned = False
        pin_direction = ()
        for i in range(len(self.pinned_pieces)-1, -1, -1):
            if self.pinned_pieces[i][0] == row and self.pinned_pieces[i][1] == col:
                piecePinned = True
                pin_direction = (self.pinned_pieces[i][2], self.pinned_pieces[i][3])
                self.pinned_pieces.remove(self.pinned_pieces[i])
                break

        if self.white_to_move:
            moveAmount = -1
            initial_pawn_row = 6
            opponent_color = 'b'
            kingRow, kingCol = self.white_king_pos
        else:
            moveAmount = 1
            initial_pawn_row = 1
            opponent_color = 'w'
            kingRow, kingCol = self.black_king_pos

        if self.board[row + moveAmount][col] == "--": # Single square move
            if not piecePinned or pin_direction == (moveAmount, 0):
                move_list.append(Move((row, col), (row + moveAmount, col), self.board))
                if row == initial_pawn_row and self.board[row + 2 * moveAmount][col] == "--": # Double square move
                    move_list.append(Move((row, col), (row + 2 * moveAmount, col), self.board))

        if col-1 >= 0: # captures to the left
            if not piecePinned or pin_direction == (moveAmount, -1):
                if self.board[row + moveAmount][col - 1][0] == opponent_color:
                    move_list.append(Move((row, col), (row + moveAmount, col - 1), self.board))
                if (row + moveAmount, col - 1) == self.en_passant_possible:
                    square_under_attack = has_blocking_piece = False
                    if kingRow == row:
                        if kingCol < col: # king is left of the pawn
                            # inside: between king and the pawn
                            # outside: between pawn and border
                            insideRange = range(kingCol + 1, col-1)
                            outsideRange = range(col + 1, 8)
                        else: # king right of the pawn
                            insideRange = range(kingCol -1, col, -1)
                            outsideRange = range(col - 2, -1, -1)
                        for i in insideRange:
                            if self.board[row][i] != "--":    # some piece beside en-passant pawn blocks
                                has_blocking_piece = True
                        for i in outsideRange:
                            square = self.board[row][i]
                            if square[0] == opponent_color and (square[1] == "R" or square[1] == "Q"):
                                square_under_attack = True
                            elif square != "--": # some piece blocks the en-passant capture (not blank square)
                                has_blocking_piece = True
                    if not square_under_attack or has_blocking_piece: # if square is not under attack or has blocking piece
                        move_list.append(Move((row, col), (row + moveAmount, col - 1), self.board, is_en_passant=True))

        if col+1 <= 7: # captures to the right
            if not piecePinned or pin_direction == (moveAmount, +1):
                if self.board[row + moveAmount][col + 1][0] == opponent_color:
                    move_list.append(Move((row, col), (row + moveAmount, col + 1), self.board))
                if (row + moveAmount, col + 1) == self.en_passant_possible:
                    square_under_attack = has_blocking_piece = False
                    if kingRow == row:
                        if kingCol < col: # king is left of the pawn
                            # inside: between king and the pawn
                            # outside: between pawn and border
                            insideRange = range(kingCol + 1, col)
                            outsideRange = range(col + 2, 8)
                        else: # king right of the pawn
                            insideRange = range(kingCol - 1, col + 1, -1)
                            outsideRange = range(col - 1, -1, -1)
                        for i in insideRange:
                            if self.board[row][i] != "--":    # some piece beside en-passant pawn blocks
                                has_blocking_piece = True
                        for i in outsideRange:
                            square = self.board[row][i]
                            if square[0] == opponent_color and (square[1] == "R" or square[1] == "Q"):
                                square_under_attack = True
                            elif square != "--":
                                has_blocking_piece = True
                    if not square_under_attack or has_blocking_piece:
                        move_list.append(Move((row, col), (row + moveAmount, col + 1), self.board, is_en_passant=True))

    def get_rook_moves(self, row, col, move_list):
        # Getting rook moves and appending them to moves list
        piecePinned = False
        pin_direction = ()
        for i in range(len(self.pinned_pieces)-1, -1, -1):
            if self.pinned_pieces[i][0] == row and self.pinned_pieces[i][1] == col:
                piecePinned = True
                pin_direction = (self.pinned_pieces[i][2], self.pinned_pieces[i][3])
                if self.board[row][col][1] != 'Q': # cannot remove queen from pin on rook moves, only remove it on bishop moves
                    self.pinned_pieces.remove(self.pinned_pieces[i])
                break

        # Get all possible moves for a rook at (row, col)) and add them to the moves list
        directions = ((1, 0), (-1, 0), (0, 1), (0, -1))  # Down, Up, Right, Left
        opponent_color = 'b' if self.white_to_move else 'w'  # Determine the opponent's color
        for direction in directions:  # Loop through each direction
            for step in range(1, 8):  # Loop through each square in the direction
                target_row = row + direction[0] * step  # Calculate the end row
                target_col = col + direction[1] * step  # Calculate the end column
                if 0 <= target_row < 8 and 0 <= target_col < 8:  # Check if the end square is within bounds
                    if not piecePinned or pin_direction == direction or pin_direction == (-direction[0], -direction[1]): # Allow movement In Pin direction or opposite to pin direction
                        target_piece = self.board[target_row][target_col]  # Get the piece at the end square
                        if target_piece == "--":  # If the square is empty
                            move_list.append(Move((row, col), (target_row, target_col), self.board))  # Add the move to the list
                        elif target_piece[0] == opponent_color:  # If the square has an opponent's piece color
                            move_list.append(Move((row, col), (target_row, target_col), self.board))  # Add the capture move to the list
                            break  # Stop checking further in this direction
                        else:  # If the square has friendly piece color
                            break  # Stop checking further in this direction
                else:   # If the end square is out of bounds
                    break   # Stop checking further in this direction

    def get_knight_moves(self, row, col, move_list):
        piecePinned = False
        for i in range(len(self.pinned_pieces)-1, -1, -1):
            if self.pinned_pieces[i][0] == row and self.pinned_pieces[i][1] == col:
                piecePinned = True
                self.pinned_pieces.remove(self.pinned_pieces[i])
                break

        # Get all possible moves for a knight at (row, col)) and add them to the moves list
        knight_offsets = ((2, 1), (2, -1), (-2, 1), (-2, -1),
                         (1, 2), (1, -2), (-1, 2), (-1, -2))
        # Knight can move in an "L" shape, so we define the possible moves
        player_color = 'w' if self.white_to_move else 'b'  # Determine the friendly color
        for move_offset in knight_offsets: # Loop through each direction
            target_row = row + move_offset[0]  # Calculate the end row
            target_col = col + move_offset[1]  # Calculate the end column
            if 0 <= target_row < 8 and 0 <= target_col < 8:  # Check if the end square is within bounds
                if not piecePinned: # if not the pinned piece then movement allowed
                    target_piece = self.board[target_row][target_col]  # Get the piece at the end square
                    if target_piece[0] != player_color:  # If the square is empty or has a enemy piece
                        move_list.append(Move((row, col), (target_row, target_col), self.board))

    def get_bishop_moves(self, row, col, move_list):
        piecePinned = False
        pin_direction = ()
        for i in range(len(self.pinned_pieces)-1, -1, -1):
            if self.pinned_pieces[i][0] == row and self.pinned_pieces[i][1] == col:
                piecePinned = True
                pin_direction = (self.pinned_pieces[i][2], self.pinned_pieces[i][3])
                self.pinned_pieces.remove(self.pinned_pieces[i])
                break

        # Get all possible moves for a bishop at (row, col)) and add them to the moves list
        directions = ((1, 1), (1, -1), (-1, 1), (-1, -1))  # Down-Right, Down-Left, Up-Right, Up-Left
        opponent_color = 'b' if self.white_to_move else 'w'  # Determine the opponent's color
        for direction in directions:  # Loop through each direction
            for step in range(1, 8):  # Loop through each square in the direction (Bishops can move 7 squares in any diagonal direction)
                target_row = row + direction[0] * step  # Calculate the end row
                target_col = col + direction[1] * step  # Calculate the end column
                if 0 <= target_row < 8 and 0 <= target_col < 8:  # Check if the end square is within bounds
                    if not piecePinned or pin_direction == direction or pin_direction == (-direction[0], -direction[1]): # Allow movement In Pin direction or opposite to pin direction
                        target_piece = self.board[target_row][target_col]  # Get the piece at the end square
                        if target_piece == "--":  # If the square is empty
                            move_list.append(Move((row, col), (target_row, target_col), self.board))  # Add the move to the list
                        elif target_piece[0] == opponent_color:  # If the square has an opponent's piece color
                            move_list.append(Move((row, col), (target_row, target_col), self.board))  # Add the capture move to the list
                            break  # Stop checking further in this direction
                        else:  # If the square has friendly piece color
                            break  # Stop checking further in this direction
                else:   # If the end square is out of bounds
                    break   # Stop checking further in this direction

    def get_queen_moves(self, row, col, move_list):
        # Get all possible moves for a queen at (row, col)) and add them to the moves list
        # A queen can move like both a rook and a bishop, so we combine their movement logic
        self.get_rook_moves(row, col, move_list)  # Use rook's movement logic
        self.get_bishop_moves(row, col, move_list)  # Use bishop's movement logic

    def get_king_moves(self, row, col, move_list):
         # Get all possible moves for a king at row and col levels
        king_row_offsets = (-1, -1, -1, 0, 0, 1, 1, 1)
        king_col_offsets = (-1, 0, 1, -1, 1, -1, 0, 1)
        player_color = 'w' if self.white_to_move else 'b'  # Determine the friencly color
        for i in range(8):  # Loop through each possible move
            target_row = row + king_row_offsets[i] # Calculate the end row
            target_col = col + king_col_offsets[i] # Calculate the end column
            if 0 <= target_row < 8 and 0 <= target_col < 8:  # Check if the end square is within bounds
                target_piece = self.board[target_row][target_col]  # Get the piece at the end square
                if target_piece[0] != player_color:  # If the square is empty or has a enemy piece
                    # Temporarily Placing king on end square and checking for checks
                    if player_color == 'w':
                        self.white_king_pos = (target_row, target_col)
                    else:
                        self.black_king_pos = (target_row, target_col)
                    inCheck, pins, checks = self.check_for_pins_and_checks()
                    if not inCheck:
                        move_list.append(Move((row, col), (target_row, target_col), self.board))  # Add the move to the list
                    # Placing king back in its original location
                    if player_color == 'w':
                        self.white_king_pos = (row, col)
                    else:
                        self.black_king_pos = (row, col)

    def get_castle_moves(self, row, col, move_list):
        # Generate all valid castle moves for the king and add them to the list of moves
        if self.is_square_under_attack(row, col):
            return # cannot castle when in check
        if (self.white_to_move and self.castling_rights_current.white_kingside) or (not self.white_to_move and self.castling_rights_current.black_kingside):
            self.get_kingside_castle_moves(row, col, move_list)
        if (self.white_to_move and self.castling_rights_current.white_queenside) or (not self.white_to_move and self.castling_rights_current.black_queenside):
            self.get_queenside_castle_moves(row, col, move_list)

    def get_kingside_castle_moves(self, row, col, move_list):
        if self.board[row][col+1] == '--' and self.board[row][col+2] == '--':
            '''
            Check for if the square is out of bounds can be made but if king is eligible for castling,
            king will be in centre and there is no point doing that check.
            '''
            if not self.is_square_under_attack(row, col+1) and not self.is_square_under_attack(row, col+2):
                move_list.append(Move((row, col), (row, col+2), self.board, is_castling_move = True))

    def get_queenside_castle_moves(self, row, col, move_list):
        if self.board[row][col-1] == '--' and self.board[row][col-2] == '--' and self.board[row][col-3] == '--':
            '''
            Check for if the square is out of bounds can be made but if king is eligible for castling,
            king will be in centre and there is no point doing that check.
            '''
            if not self.is_square_under_attack(row, col-1) and not self.is_square_under_attack(row, col-2):
                move_list.append(Move((row, col), (row, col-2), self.board, is_castling_move = True))

'''
Class storing current state of Castling rights
'''
class CastleRights():
    def __init__(self, white_kingside, black_kingside, white_queenside, black_queenside):
        self.white_kingside = white_kingside
        self.black_kingside = black_kingside
        self.white_queenside = white_queenside
        self.black_queenside = black_queenside

'''
Class to store movement data (Storing move log and current snapshot of board)
'''
class Move():

    ranksToRows = {"1": 7, "2": 6, "3": 5, "4": 4, "5": 3, "6": 2, "7": 1, "8": 0}  # Maps chess ranks (as strings) to board row indices
    rowsToRanks = {v: k for k,
                   v in ranksToRows.items()}    # Inverse mapping: board row indices to chess ranks
    filesToCols = {"h": 7, "g": 6, "f": 5, "e": 4, "d": 3, "c": 2, "b": 1, "a": 0}  # Maps chess files (as strings) to board column indices
    colsToFiles = {v: k for k,
                   v in filesToCols.items()}    # Inverse mapping: board col indices to chess files

    def __init__(self, start_square, end_square, board, is_en_passant = False, is_castling_move = False):
        # Start pos, end pos and board layout
        self.start_row = start_square[0]  # Starting Row pos
        self.start_col = start_square[1]  # Starting Col pos
        self.end_row = end_square[0]  # Ending Row pos
        self.end_col = end_square[1]  # Ending Col pos
        self.moved_piece = board[self.start_row][self.start_col]   # Piece that is being moved
        self.captured_piece = board[self.end_row][self.end_col]    # Piece being captured
        self.is_en_passant = is_en_passant
        self.is_pawn_promotion = (self.moved_piece == 'wp' and self.end_row == 0) or \
            (self.moved_piece == 'bp' and self.end_row == 7)   # Pawn promotion flag
        self.is_castling_move = is_castling_move
        if is_en_passant:
            # en-passant only captures opposite colored pawn
            self.captured_piece = 'wp' if self.moved_piece == 'bp' else 'bp'
        self.is_captured = self.captured_piece != '--' # If a piece is captured, it is not an empty square
        # Unique ID for the move based on start and end positions
        self.unique_id = self.start_row * 1000 + self.start_col * 100 + self.end_row * 10 + self.end_col


    def __eq__(self, other):
        # Check if two Move objects are equal based on their start and end positions
        if isinstance(other, Move):
            # Compare the moveID of both Move objects
            return self.unique_id == other.unique_id  # Unique ID comparison for equality
        # If other is not a Move instance, return False
        return False

    def get_chess_notation(self):
        # Returns a string representation of the move in chess notation
        return self.get_rank_file(self.start_row, self.start_col) + self.get_rank_file(self.end_row, self.end_col)

    def get_rank_file(self, row, col):
        # Converts board row and column indices to chess notation (e.g., (0, 0) -> 'a8')
        return self.colsToFiles[col] + self.rowsToRanks[row]
    
    """
    Overriding string representation of the move in chess notation.
    """
    def __str__(self):
        
        # Castling move
        if self.is_castling_move:
            return "O-O" if self.end_col == 6 else "O-O-O"
        end_square = self.get_rank_file(self.end_row, self.end_col) # Get the end square in chess notation
        
        # Pawn moves
        if self.moved_piece[1] == 'p':
            if self.is_captured:
                return self.colsToFiles[self.start_col] + 'x' + end_square
            else:
                return end_square

        # Other piece moves
        move_string = self.moved_piece[1] # Get the piece type
        if self.is_captured: # If a piece is captured, add 'x' to the move string
            move_string += 'x' 
        return move_string + end_square  # Return the move string in chess notation