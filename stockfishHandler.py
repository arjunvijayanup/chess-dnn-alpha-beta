"""
This module provides a robust interface for the Stockfish chess engine.
It encapsulates the logic for locating, initializing and communicating with the Stockfish executable,
allowing the main game loop to request moves without handling the low-level details.
Stockfish was downloaded from https://stockfishchess.org/download/
"""
import chess
import chess.engine
import os
import platform
import shutil
import chessEngine

class StockfishPlayer:
    """
    A class to handle all interactions with the Stockfish chess engine.
    It encapsulates the engine's initialization, move generation, and termination.
    """
    def __init__(self, base_dir, depth):
        # The engine is initialized only once when a StockfishPlayer object is created.
        self.stockfish_engine = self._setup_stockfish_engine(base_dir)
        self.depth = depth

    def _setup_stockfish_engine(self, base_dir):
        # Set up Stockfish engine path based on the operating system.
        system = platform.system()
        if system == "Windows":
            STOCKFISH_PATH = os.path.join(base_dir, "stockfish/stockfish-windows-x86-64-avx2.exe")
        elif system == "Darwin":
            # macOS (Apple-Silicon chip)
            STOCKFISH_PATH = os.path.join(base_dir, "stockfish/stockfish-macos-m1-apple-silicon")
        else:
            raise RuntimeError(f"Unsupported OS: {system!r}")

        # Fallback if the default path doesn't work.
        if not os.path.isfile(STOCKFISH_PATH) or not os.access(STOCKFISH_PATH, os.X_OK):
            fallback = shutil.which("stockfish")
            if fallback:
                STOCKFISH_PATH = fallback
            else:
                raise FileNotFoundError(f"Could not find Stockfish binary at {STOCKFISH_PATH!r}, nor on your PATH.")
        
        # Open the engine as a subprocess.
        return chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    """
     Calculates and returns the best move from the Stockfish engine.    
    """
    def get_best_move(self, game_state):

        # Convert the game state to FEN format for Stockfish
        fen = game_state.to_fen()
        board = chess.Board(fen)
        # Get the best move from the engine with the specified search depth.
        result = self.stockfish_engine.play(board, chess.engine.Limit(depth=self.depth))
        # Convert the move from Stockfish's format to your game's format.
        sfm = result.move
        start_row = 7 - (sfm.from_square // 8)
        start_col = sfm.from_square % 8
        end_row = 7 - (sfm.to_square // 8)
        end_col = sfm.to_square % 8
        
        return chessEngine.Move((start_row, start_col), (end_row, end_col), game_state.board)

    """
    Gracefully terminate the Stockfish engine process.
    """
    def quit_engine(self):

        try:
            self.stockfish_engine.quit()
        except (chess.engine.EngineTerminatedError, chess.engine.EngineError, BrokenPipeError):
            pass