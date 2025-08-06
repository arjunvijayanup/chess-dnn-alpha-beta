"""
Main driver file for the Chess game. Responsible for handling user input and displaying current GameState object
"""
import pygame as p
import ChessEngine, ChessAI
from multiprocessing import Process, Queue

BOARD_WIDTH = BOARD_HEIGHT = 512
MOVE_LOG_SECTION_WIDTH = 300
MOVE_LOG_SECTION_HEIGHT = BOARD_HEIGHT
DIMENSION = 8
SQ_SIZE = BOARD_HEIGHT // DIMENSION
MAX_FPS = 20
IMAGES = {}

"""
Initializing global dict of images. Called once in the main
"""
def load_images():
    pieces = ['wp', 'bp', 'wB', 'wK', 'wQ', 'wN', 'wR', 'bB', 'bK', 'bQ', 'bN', 'bR']
    for piece in pieces:
        IMAGES[piece] = p.transform.scale(p.image.load("images_pieces/" + piece + ".png"), (SQ_SIZE, SQ_SIZE)) # Used transform.scale to make the pieces look good on the board.
# Use IMAGES['wp'] to load images

"""
Initializing main. Handles user input and updating graphics
"""
def main():
    p.init() # Initialize the pygame module
    screen = p.display.set_mode((BOARD_WIDTH + MOVE_LOG_SECTION_WIDTH, BOARD_HEIGHT)) # Set the window size
    clock = p.time.Clock() # Create a clock object
    game_state = ChessEngine.GameState() # Create a GameState object calling the Gamestate __init_ method
    legal_moves = game_state.get_valid_moves() # Get the valid moves for the current game state
    move_executed = False # Flag Variable to check if a move has been made
    should_animate = False # Flag variable to check when to animate a move
    move_log_font = p.font.SysFont("Arial", 12, False, False) # Font for the move log (arial 11 size, not bold, not italic)
    endgame_text_font = p.font.SysFont("Helvitca", 32, True, False) # Bold, not italic, 32 size, Helvetica

    p.display.set_caption("Chess") # Set the window title
    load_images() # Load the images for the pieces before the while loop
    is_running = True # Variable to control the main loop
    selected_square = () # Tuple (row,col) to store the last click of square selected by the user
    click_history = [] # List to store the clicks made by the player (two tuples: [(row, col), (row, col)])
    is_game_over = False # Flag for when game ends
    promotion_pending_move = None
    human_white_player = False # Flag for white player (True if human, False if AI)
    human_black_player = False # Flag for black player (True if human, False if AI)
    AI_thinking = False # Flag for AI thinking (True if AI is thinking, False if not)
    AI_thinking_process = None # Process for AI move finding
    move_undone = False # Flag for move undoing (True if move is undone, False if not)

    while is_running:
        clock.tick(MAX_FPS)
        # If human player is playing, set human_turn to True
        human_turn = (game_state.white_to_move and human_white_player) or (not game_state.white_to_move and human_black_player)

        for event in p.event.get():
            if event.type == p.QUIT: # Check if the user wants to quit
                is_running = False # Set running to False to exit the loop
            elif event.type == p.MOUSEBUTTONDOWN: # Check if the user clicked the mouse
                if not is_game_over and not promotion_pending_move: # If the game is not over, no promotion pending, process the mouse click
                    click_position = p.mouse.get_pos() # Get the mouse (x,y) position
                    clicked_col = click_position[0] // SQ_SIZE # Calculate the column based on mouse position
                    clicked_row = click_position[1] // SQ_SIZE # Calculate the row based on mouse position
                    
                    if selected_square == (clicked_row, clicked_col) or clicked_col >= 8: # If the same square is clicked twice or clicked outside the board (i.e move log section)
                        selected_square, click_history = (), [] # deselect the selected square
                    else:
                        selected_square = (clicked_row, clicked_col) # Update the selected square
                        click_history.append(selected_square) # Add the selected square to player clicks
                    
                    if len(click_history) == 2 and human_turn: # Process move if two squares are selected and it's human's turn (after 2nd click)
                        move = ChessEngine.Move(click_history[0], click_history[1], game_state.board)
                        print(move.get_chess_notation()) # Print the move in chess notation (for debugging purposes)
                        move_found = False
                        for move_index in range(len(legal_moves)):
                            if move == legal_moves[move_index]:
                                move_to_make = legal_moves[move_index]
                                if move_to_make.is_pawn_promotion:
                                    promotion_pending_move = move_to_make
                                    print("PAWN PROMOTION: Press 'Q', 'R', 'B', or 'N'.")
                                else:
                                    game_state.make_move(move_to_make) # Make the move in the game state
                                    move_executed, should_animate = True, True
                                    selected_square, click_history = (), [] # Reset the player clicks after making the move
                                move_found = True
                                break
                        if not move_found: # If the move is not valid
                            click_history = [selected_square]   # Reset the player clicks to only the last selected square

            elif event.type == p.KEYDOWN: # Check if a key is pressed
                if promotion_pending_move:
                    promotion_choice = None
                    if event.key == p.K_q: promotion_choice = 'Q'
                    elif event.key == p.K_r: promotion_choice = 'R'
                    elif event.key == p.K_b: promotion_choice = 'B'
                    elif event.key == p.K_n: promotion_choice = 'N'
                    
                    if promotion_choice:
                        game_state.make_move(promotion_pending_move, promotion_choice)
                        move_executed, should_animate = True, True
                        promotion_pending_move, selected_square, click_history = None, (), []
                else:
                    if event.key == p.K_z: # If 'z' key is pressed 
                        game_state.undo_move() # Undo the last move made in the game state
                        move_executed, should_animate, is_game_over = True, False, False
                        legal_moves = game_state.get_valid_moves()
                        selected_square, click_history = (), [] # Undo any square selections
                        if AI_thinking:
                            AI_thinking_process.terminate() # Terminate the AI move finding process if it is running
                            AI_thinking = False # Reset the AI thinking flag if AI was thinking
                        move_undone = True # Set the move undone flag to True
                    if event.key == p.K_r: # if 'r' key is pressed
                        game_state = ChessEngine.GameState() # Reinitiating Gamestate
                        legal_moves = game_state.get_valid_moves()
                        selected_square, click_history = (), [] # Undo any square selections
                        move_executed, should_animate, is_game_over = False, False, False
                        move_undone = False # Reset the move undone flag
        
        # AI move logic
        if not is_game_over and not promotion_pending_move and not human_turn and not move_undone: # If the game is not over, no promotion pending, and it's the AI's turn
            if not AI_thinking:
                AI_thinking = True
                print("AI is thinking...") # Print AI is thinking message
                return_queue = Queue() # queue to track the best move from the AI within each thread and pass data between threads
                AI_thinking_process = Process(target=ChessAI.get_best_move, args=(game_state, legal_moves, return_queue))
                AI_thinking_process.start() # Start the AI move finding process
            if not AI_thinking_process.is_alive(): # If the AI move finding process is still running
                print("AI finished thinking.") # Print AI finished thinking message
                if not return_queue.empty(): # If the return queue is not empty
                    AI_move = return_queue.get() # Get the best move from the AI
                else:
                    AI_move = None # If the return queue is empty, set AI_move to None
                if AI_move is None: # If best move is none
                    AI_move = ChessAI.random_AI_move(legal_moves) # Get a random move from the Random AI function
                game_state.make_move(AI_move) # Make the AI move in the game state
                move_executed, should_animate = True, True # Set the flags to indicate a move has been made
                AI_thinking = False # Reset the AI thinking flag

        
        if move_executed: # If a move has been made
            if should_animate: animate_move(game_state.moves_log[-1], screen, game_state.board, clock) # taking latest move and animating
            legal_moves = game_state.get_valid_moves() # Get the valid moves for the current game state
            
            # Setting check and checkmate flags for the last move in move log (for Move log panel Chess Notation)
            if game_state.moves_log: # If there are moves in the move log
                last_move = game_state.moves_log[-1]
                if game_state.is_in_check: # If the last move resulted in check
                    last_move.is_in_check = True
                if game_state.is_checkmate: # If the last move resulted in checkmate
                    last_move.is_checkmate = True
            
            move_executed, should_animate = False, False # Reset the flags
            move_undone = False # Reset the move undone flag

        draw_game_state(screen, game_state, legal_moves, selected_square, promotion_pending_move, move_log_font)
        
        if game_state.is_checkmate or game_state.is_stalemate : # If checkmated, gameover and print checkmate message
            is_game_over = True
            if game_state.is_stalemate: 
                print_text = 'Stalemate!' 
            else: 
                print_text = 'Black wins by Checkmate!' if game_state.white_to_move else 'White wins by Checkmate!'
            draw_endgame_text(screen, print_text, endgame_text_font) # Draw the endgame text on the screen
        p.display.flip() # Update the display

"""
Graphics in the current game state
"""
def draw_game_state(screen, game_state, legal_moves, selected_square, promotion_pending_move, move_log_font):
    draw_board(screen) # Draw the squares on board
    highlight_squares(screen, game_state, legal_moves, selected_square, promotion_pending_move) # Highlight current chess piece and its valid moves
    draw_pieces(screen, game_state.board) # Draw the pieces on the squares on board
    draw_move_log(screen, game_state, move_log_font) # Draw the move log text on the screen

"""
Draws the squares on the chess board.
"""
def draw_board(screen):
    colors = [p.color.Color("lightyellow"), p.color.Color("darkolivegreen")]  # Define the colors for the squares
    for row in range(DIMENSION):  # Loop through each row
        for col in range(DIMENSION): # Loop through each column
            color = colors[((row + col) % 2)]  # Alternate colors for squares
            p.draw.rect(screen, color, p.Rect(col * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE))  # Draw the square

"""
Highlighting the selected square and promotion choice
"""
def highlight_squares(screen, game_state, legal_moves, selected_square, promotion_pending_move):
    # highlight king when in check
    if game_state.is_in_check:
        king_pos = game_state.white_king_pos if game_state.white_to_move else game_state.black_king_pos
        s = p.Surface((SQ_SIZE, SQ_SIZE))
        s.set_alpha(120)  # semi-transparent
        s.fill(p.Color('red'))
        screen.blit(s, (king_pos[1] * SQ_SIZE, king_pos[0] * SQ_SIZE))

    if promotion_pending_move:
        s = p.Surface((SQ_SIZE, SQ_SIZE))
        s.set_alpha(150)
        s.fill(p.Color('blue'))
        screen.blit(s, (promotion_pending_move.end_col * SQ_SIZE, promotion_pending_move.end_row * SQ_SIZE))
        return

    if selected_square != ():
        row, col = selected_square
        if game_state.board[row][col][0] == ('w' if game_state.white_to_move else 'b'): #if sqSelected is a piece that can be moved
            # Highlight the selected square
            s = p.Surface((SQ_SIZE, SQ_SIZE)) # Create a surface for highlighting
            s.set_alpha(100) # Setting transparency level 100 is fully opaque
            s.fill(p.Color("orange")) # Fill the surface with orange color
            screen.blit(s, (col * SQ_SIZE, row * SQ_SIZE)) # Draw the highlight on the selected square
            s.fill(p.Color('orange')) # Fill valid move surfaces to orange color
            for move in legal_moves:
                if move.start_row == row and move.start_col == col:
                    # Draw the highlight on the selected valid movement squares
                    screen.blit(s, (move.end_col * SQ_SIZE, move.end_row * SQ_SIZE))

"""
Draws the pieces on the chess board
"""
def draw_pieces(screen, board):
    for row in range(DIMENSION):  # Loop through each row
        for col in range(DIMENSION):  # Loop through each column
            piece = board[row][col]  # Get the piece at the current square
            if piece != "--":  # If there is a piece on the square (not empty)
                screen.blit(IMAGES[piece], p.Rect(col * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE))  # Draw the piece

"""
Draws the move log text on the screen
"""
def draw_move_log(screen, game_state, font):
    move_log_section_rect = p.Rect(BOARD_WIDTH, 0, MOVE_LOG_SECTION_WIDTH, MOVE_LOG_SECTION_HEIGHT) # Create a rectangle for the move log section
    p.draw.rect(screen, p.Color('grey'), move_log_section_rect)  # Draw the white section for the move log
    p.draw.rect(screen, p.Color('darkgrey'), move_log_section_rect, 2)  # Draw the border for the move log section
    move_log = game_state.moves_log  # Get the move log from the game state
    move_texts = []  # List to store the text for each move
    move_text_x_padding = move_text_y_padding = 5 # Initial Padding for the text in the move log section
    move_text_line_space = 2  # Space between lines of text
    moves_per_row = 3 # Number of moves per row in the move log section

    """
    Loop through the move log in pairs (white and black moves)
    Storing the moves in pairs for better readability
    For every two moves, one for white and one for black
    """
    for i in range(0, len(move_log), 2):  # Loop through the move log in pairs
        move_string = str(i // 2 + 1) + '. ' + str(move_log[i]) # Get the white move notation, with move number
        if i + 1 < len(move_log):  # If there is a black move
            move_string += ' ' + str(move_log[i + 1]) # Get the black move notation
        move_texts.append(move_string)  # Add the move string to the move texts list

    # Loop through the move texts rows and display them on the screen
    for i in range(0, len(move_texts), moves_per_row):
        move_text = "" # Initialize the move text for the current row
        for j in range(moves_per_row):  # Loop through the moves per row 
            if i + j < len(move_texts):  # Check if the index is within bounds
                move_text += move_texts[i+j] + "    "   # Concatenate the move text for the current row
        text_surface = font.render(move_text, True, p.Color("black"))  # Render the text with black color
        text_rect = move_log_section_rect.move(move_text_x_padding, move_text_y_padding) # Create a location for the text surface
        screen.blit(text_surface, text_rect) # Display the text in the specified location
        move_text_y_padding += text_surface.get_height() + move_text_line_space  # Update the padding for the next move text

"""
Animating chess piece movement
"""
def animate_move(move, screen, board, clock):
    colors = [p.color.Color("lightyellow"), p.color.Color("darkolivegreen")]
    row_delta, col_delta = move.end_row - move.start_row, move.end_col - move.start_col # Row and column change
    frames_per_square = 5 # Frames per square
    frame_count = (abs(row_delta) + abs(col_delta)) * frames_per_square  # total frame count b/w rows and cols
    for frame in range(frame_count + 1): # Going through each frame
        row, col = (move.start_row + row_delta * frame / frame_count, move.start_col + col_delta * frame / frame_count)
        draw_board(screen)
        draw_pieces(screen, board)
        
        # Erasing piece moved from ending square (As it is already handled in GameState - avoiding duplicate piece and endSquare)
        color = colors[(move.end_row + move.end_col) % 2]
        end_square = p.Rect(move.end_col * SQ_SIZE, move.end_row * SQ_SIZE, SQ_SIZE, SQ_SIZE)
        p.draw.rect(screen, color, end_square) # Replacing piece with square color
        
        # Draw the captured piece on top of the rectangle drawn
        if move.captured_piece != '--': 
            if move.is_en_passant:  # If the move is an en-passant move
                en_passant_row = move.end_row - 1 if move.moved_piece[0] == 'w' else move.end_row + 1   # The row where the captured (en-passanted) pawn was
                end_square = p.Rect(move.end_col * SQ_SIZE, en_passant_row * SQ_SIZE, SQ_SIZE, SQ_SIZE)    # Draw the square where the captured (en-passanted) pawn was
        # Drawing moving piece
        screen.blit(IMAGES[move.moved_piece], p.Rect(col * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE))
        p.display.flip()
        clock.tick(60) # 60 fps (60 frames per sec - for one square movement 1/6 sec)

"""
Draws text for game over messages
"""
def draw_endgame_text(screen, text, font):
    text_surface = font.render(text, 0, p.Color('Gray')) # Render in black color with no aliasing
    text_rect = p.Rect(0, 0, BOARD_WIDTH, BOARD_HEIGHT).move(BOARD_WIDTH/2 - text_surface.get_width()/2, 
                                                                 BOARD_HEIGHT/2 - text_surface.get_height()/2) # Centering the text on the board
    screen.blit(text_surface, text_rect)
    text_surface = font.render(text, 0, p.Color("Black"))
    screen.blit(text_surface, text_rect.move(2, 2))

if __name__ == "__main__":
    main() # Call the main function to start the game (Only runs main if this file is run directly)