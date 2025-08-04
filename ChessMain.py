"""
Main driver file for the Chess game. Responsible for handling user input and displaying current GameState object
"""
import pygame as p
import ChessEngine, ChessAI

WIDTH = 512 
HEIGHT = 512
HEADER_HEIGHT = 30 # bar for the timer
DIMENSION = 8
SQ_SIZE = HEIGHT // DIMENSION
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
    screen = p.display.set_mode((WIDTH, HEIGHT + HEADER_HEIGHT)) # Set the window size
    clock = p.time.Clock() # Create a clock object
    game_state = ChessEngine.GameState() # Create a GameState object calling the Gamestate __init_ method
    legal_moves = game_state.get_valid_moves() # Get the valid moves for the current game state
    move_executed = False # Flag Variable to check if a move has been made
    should_animate = False # Flag variable to check when to animate a move

    p.display.set_caption("Chess") # Set the window title
    load_images() # Load the images for the pieces before the while loop
    is_running = True # Variable to control the main loop
    selected_square = () # Tuple (row,col) to store the last click of square selected by the user
    click_history = [] # List to store the clicks made by the player (two tuples: [(row, col), (row, col)])
    is_game_over = False # Flag for when game ends
    promotion_pending_move = None
    human_white_player = False # Flag for white player (True if human, False if AI)
    human_black_player = False # Flag for black player (True if human, False if AI)
    # timer bar
    white_time_seconds = 300  # 5 minutes
    black_time_seconds = 300  # 5 minutes
    timer_font = p.font.SysFont("Helvitca", 20, True, False)

    while is_running:
        time_delta = clock.tick(MAX_FPS) / 1000.0
        if not is_game_over and not promotion_pending_move:
            if game_state.white_to_move:
                white_time_seconds -= time_delta
            else:
                black_time_seconds -= time_delta
        # If human player is playing, set human_turn to True
        human_turn = (game_state.white_to_move and human_white_player) or (not game_state.white_to_move and human_black_player)

        for event in p.event.get():
            if event.type == p.QUIT: # Check if the user wants to quit
                is_running = False # Set running to False to exit the loop
            elif event.type == p.MOUSEBUTTONDOWN: # Check if the user clicked the mouse
                if not is_game_over and not promotion_pending_move and human_turn: # If the game is not over, no promotion pending, and it's the human player's turn
                    click_position = p.mouse.get_pos() # Get the mouse (x,y) position
                    # Adjust click position for the header
                    if click_position[1] < HEADER_HEIGHT: continue
                    clicked_col = click_position[0] // SQ_SIZE # Calculate the column based on mouse position
                    clicked_row = (click_position[1] - HEADER_HEIGHT) // SQ_SIZE # Calculate the row based on mouse position
                    
                    if selected_square == (clicked_row, clicked_col): # If the same square is clicked twice
                        selected_square, click_history = (), [] # deselect the selected square
                    else:
                        selected_square = (clicked_row, clicked_col) # Update the selected square
                        click_history.append(selected_square) # Add the selected square to player clicks
                    
                    if len(click_history) == 2: # If two squares are selected (After the 2nd click)
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
                    if event.key == p.K_r: # if 'r' key is pressed
                        game_state = ChessEngine.GameState() # Reinitiating Gamestate
                        legal_moves = game_state.get_valid_moves()
                        selected_square, click_history = (), [] # Undo any square selections
                        move_executed, should_animate, is_game_over = False, False, False
                        white_time_seconds, black_time_seconds = 300, 300

        # AI move logic
        if not is_game_over and not promotion_pending_move and not human_turn: # If the game is not over, no promotion pending, and it's the AI's turn
            AI_move = ChessAI.get_best_minimax_move(game_state, legal_moves) # Get the best move from the Best Move AI function
            if AI_move is None: # If best move is none
                AI_move = ChessAI.random_AI_move(legal_moves) # Get a random move from the Random AI function
            game_state.make_move(AI_move) # Make the AI move in the game state
            move_executed, should_animate = True, True # Set the flags to indicate a move has been made
        
        if move_executed: # If a move has been made
            if should_animate: animate_move(game_state.moves_log[-1], screen, game_state.board, clock) # taking latest move and animating
            legal_moves = game_state.get_valid_moves() # Get the valid moves for the current game state
            move_executed, should_animate = False, False # Reset the flags

        # Pass timer variables to drawing function
        draw_game_state(screen, game_state, legal_moves, selected_square, promotion_pending_move, white_time_seconds, black_time_seconds, timer_font)
        
        if game_state.is_checkmate: # If checkmated, gameover and print checkmate message
            is_game_over = True
            if game_state.white_to_move: draw_text(screen, 'Black wins by Checkmate!')
            else: draw_text(screen, 'White wins by Checkmate!')
        elif game_state.is_stalemate:
            is_game_over = True
            draw_text(screen, 'Stalemate!')
        elif white_time_seconds <= 0:
            is_game_over = True
            draw_text(screen, 'Black wins on time!')
        elif black_time_seconds <= 0:
            is_game_over = True
            draw_text(screen, 'White wins on time!')

        p.display.flip() # Update the display

"""
Graphics in the current game state
"""
# Accept timer variables to pass them along to the timer drawing function
def draw_game_state(screen, game_state, legal_moves, selected_square, promotion_pending_move, white_time, black_time, font):
    p.draw.rect(screen, p.Color("gainsboro"), p.Rect(0, 0, WIDTH, HEADER_HEIGHT)) # Draw header background
    draw_board(screen) # Draw the squares on board
    highlight_squares(screen, game_state, legal_moves, selected_square, promotion_pending_move) # Highlight current chess piece and its valid moves
    draw_pieces(screen, game_state.board) # Draw the pieces on the squares on board
    draw_timer(screen, font, white_time, black_time, game_state.white_to_move) # Draw the timer

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
        screen.blit(s, (king_pos[1] * SQ_SIZE, king_pos[0] * SQ_SIZE + HEADER_HEIGHT))

    if promotion_pending_move:
        s = p.Surface((SQ_SIZE, SQ_SIZE))
        s.set_alpha(150)
        s.fill(p.Color('blue'))
        screen.blit(s, (promotion_pending_move.end_col * SQ_SIZE, promotion_pending_move.end_row * SQ_SIZE + HEADER_HEIGHT))
        return

    if selected_square != ():
        row, col = selected_square
        if game_state.board[row][col][0] == ('w' if game_state.white_to_move else 'b'): #if sqSelected is a piece that can be moved
            # Highlight the selected square
            s = p.Surface((SQ_SIZE, SQ_SIZE)) # Create a surface for highlighting
            s.set_alpha(100) # Setting transparency level 100 is fully opaque
            s.fill(p.Color("orange")) # Fill the surface with orange color
            screen.blit(s, (col * SQ_SIZE, row * SQ_SIZE + HEADER_HEIGHT)) # Draw the highlight on the selected square
            s.fill(p.Color('orange')) # Fill valid move surfaces to orange color
            for move in legal_moves:
                if move.start_row == row and move.start_col == col:
                    # Draw the highlight on the selected valid movement squares
                    screen.blit(s, (move.end_col * SQ_SIZE, move.end_row * SQ_SIZE + HEADER_HEIGHT))

"""
Draws the squares on the chess board.
"""
def draw_board(screen):
    colors = [p.color.Color("lightyellow"), p.color.Color("darkolivegreen")]  # Define the colors for the squares
    for row in range(DIMENSION):  # Loop through each row
        for col in range(DIMENSION): # Loop through each column
            color = colors[((row + col) % 2)]  # Alternate colors for squares
            p.draw.rect(screen, color, p.Rect(col * SQ_SIZE, row * SQ_SIZE + HEADER_HEIGHT, SQ_SIZE, SQ_SIZE))  # Draw the square

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
        p.draw.rect(screen, p.Color("gainsboro"), p.Rect(0, 0, WIDTH, HEADER_HEIGHT))
        draw_board(screen)
        draw_pieces(screen, board)
        
        # Erasing piece moved from ending square (As it is already handled in GameState - avoiding duplicate piece and endSquare)
        color = colors[(move.end_row + move.end_col) % 2]
        end_square = p.Rect(move.end_col * SQ_SIZE, move.end_row * SQ_SIZE + HEADER_HEIGHT, SQ_SIZE, SQ_SIZE)
        p.draw.rect(screen, color, end_square) # Replacing piece with square color
        
        # Draw the captured piece on top of the rectangle drawn
        if move.captured_piece != '--': screen.blit(IMAGES[move.captured_piece], end_square)
        
        # Drawing moving piece
        screen.blit(IMAGES[move.moved_piece], p.Rect(col * SQ_SIZE, row * SQ_SIZE + HEADER_HEIGHT, SQ_SIZE, SQ_SIZE))
        p.display.flip()
        clock.tick(60) # 60 fps (60 frames per sec - for one square movement 1/6 sec)

"""
Draws the pieces on the chess board
"""
def draw_pieces(screen, board):
    for row in range(DIMENSION):  # Loop through each row
        for col in range(DIMENSION):  # Loop through each column
            piece = board[row][col]  # Get the piece at the current square
            if piece != "--":  # If there is a piece on the square (not empty)
                screen.blit(IMAGES[piece], p.Rect(col * SQ_SIZE, row * SQ_SIZE + HEADER_HEIGHT, SQ_SIZE, SQ_SIZE))  # Draw the piece

"""
Draws the timer at the top of the window
"""
def draw_timer(screen, font, white_time, black_time, white_to_move):
    white_time = max(0, white_time)
    black_time = max(0, black_time)
    
    white_mins, white_secs = divmod(white_time, 60)
    black_mins, black_secs = divmod(black_time, 60)
    
    white_str = f"{int(white_mins):02}:{int(white_secs):02}"
    black_str = f"{int(black_mins):02}:{int(black_secs):02}"
    
    white_color = p.Color('darkgreen') if white_to_move else p.Color('black') # Highlight the active player's timer
    black_color = p.Color('darkgreen') if not white_to_move else p.Color('black')

    white_text = font.render(white_str, True, white_color)
    black_text = font.render(black_str, True, black_color)

    # Position timers
    screen.blit(white_text, white_text.get_rect(midleft=(10, HEADER_HEIGHT // 2)))
    screen.blit(black_text, black_text.get_rect(midright=(WIDTH - 10, HEADER_HEIGHT // 2)))

"""
Draws text for game over messages
"""
def draw_text(screen, text):
    font = p.font.SysFont("Helvitca", 32, True, False) # Bold, not italic, 32 size, Helvetica
    text_surface = font.render(text, 0, p.Color('Gray')) # Render in black color with no aliasing
    text_rect = p.Rect(0, 0, WIDTH, HEIGHT + HEADER_HEIGHT).move(WIDTH/2 - text_surface.get_width()/2, 
                                                    (HEIGHT + HEADER_HEIGHT)/2 - text_surface.get_height()/2) # Centering the text on the board
    screen.blit(text_surface, text_rect)
    text_surface = font.render(text, 0, p.Color("Black"))
    screen.blit(text_surface, text_rect.move(2, 2))

if __name__ == "__main__":
    main() # Call the main function to start the game (Only runs main if this file is run directly)