# openingBook.py

from __future__ import annotations 
import gzip # for compressing/decompressing cache files
import pickle # for serializing/deserializing Python objects (caching)
import random # for temperature-weighted random move selection
from typing import Dict, List, Tuple, Optional # for type hints
from datasets import load_dataset # for loading Hugging Face datasets

FILES = "abcdefgh" # Chess file names for UCI notation

# Converts board coordinates (start row, start col, end row, end col) and an optional promotion piece to UCI string format.
def _idx_to_uci(start_row:int, start_col:int, end_row:int, end_col:int, promo:Optional[str]=None) -> str:
    u = f"{FILES[start_col]}{8 - start_row}{FILES[end_col]}{8 - end_row}" # Format the move using file and rank notation
    if promo: # If there is a promotion piece
        u += promo.lower() # Append the lowercase promotion piece to the UCI string
    return u # Return the UCI string

# Converts a move object to Universal Chess Interface (UCI) string format.
def _move_to_uci(move) -> str:
    promo = getattr(move, "promotion_choice", None) # Get promotion choice if available
    return _idx_to_uci(move.start_row, move.start_col, move.end_row, move.end_col, promo) # Convert move coordinates to UCI notation

# Converts a list of move objects (moves_log) to a tuple of UCI strings.
# Converts a list of moves to a tuple of UCI strings
def _moves_log_to_uci(moves_log:List) -> Tuple[str, ...]:
    return tuple(_move_to_uci(m) for m in moves_log) # Apply _move_to_uci to each move in the log

class OpeningBook:
    """
    Builds a prefix dictionary with prefix as key values and candidates as tuple values
        prefix: Tuple[uci,...]  -->  { next_uci_move: weight, ... }
    """
    def __init__(
        self,
        hf_name: str = "Lichess/chess-openings",                        # Hugging Face dataset name (default "Lichess/chess-openings")
        split: str = "train",                                           # HF dataset split (default "train")
        cache_file: Optional[str] = "opening_moveCache.pkl.gz",         # cache storage (for subsequent run speedups)
        force_rebuild: bool = False,                                    # If True, ignore cache and rebuild from HF dataset.
        limit: Optional[int] = None,                                    # Limit number of rows from HF Dataset
        max_book_plies: int = 20,                                       # Index limit for number of plies per line
        temperature: float = 0.75,                                      # Softmax temp when sampling across multiple opening book moves
        random_seed: Optional[int] = None,                              # If set, makes opening book-picking consistent across runs.
        verbose: bool = False                                           # Output description style
    ):
        self.hf_name = hf_name
        self.split = split
        self.cache_file = cache_file
        self.force_rebuild = force_rebuild
        self.limit = limit
        self.max_book_plies = max(0, int(max_book_plies))
        self.temperature = max(0.01, float(temperature))
        self.rng = random.Random(random_seed) if random_seed is not None else random
        self.verbose = verbose

        self._prefix: Dict[Tuple[str, ...], Dict[str, int]] = {}
        self.enabled = False

        try: 
            # Attempt to load the opening book from a cache file if it exists and rebuilding is not forced
            if (not force_rebuild) and cache_file and self._try_load_cache(cache_file):
                self.enabled = True
            else:
                # If cache is not used or loading fails, build the book from the Hugging Face dataset
                self._build_from_hf(self.hf_name, self.split, self.limit)
                if cache_file:
                    # Save the newly built book to cache if a cache file is specified
                    self._try_save_cache(cache_file)
                # Enable the book if any prefixes were successfully loaded or built
                self.enabled = True if self._prefix else False
        finally:
            # Final status message is always printed when verbose = true
            if self.verbose:
                total_edges = sum(len(v) for v in self._prefix.values())
                print(f"[OpeningBook] enabled={self.enabled} prefixes={len(self._prefix)} edges={total_edges}")
    

    """
    Public API methods for the OpeningBook class
    """
    def pick(self, game_state, legal_moves: List) -> Optional[object]:
        """
        Return a book move (one of `legal_moves`) or None if out-of-book.
        """
        if not self.enabled:
            return None

        # Get the current ply (half-move number) from the game state
        ply = len(getattr(game_state, "moves_log", ()))
        # If the current ply exceeds the maximum book plies, return None (out of book)
        if ply >= self.max_book_plies:
            return None

        # Convert the current game's move history to UCI format to form the prefix (notation)
        prefix = _moves_log_to_uci(game_state.moves_log)
        # Look up the prefix (notation) in the opening book to get candidate next moves
        bucket = self._prefix.get(prefix)
        # If no moves are found for the current prefix, return None (out of book)
        if not bucket:
            return None

        # Map legal moves in UCI format
        legal_by_uci = { _move_to_uci(move): move for move in legal_moves }

        # Filter book move selections to only include moves that are currently legal in the game
        candidates = [(uci, wt) for uci, wt in bucket.items() if uci in legal_by_uci]
        # If no legal book moves are found, return None
        if not candidates:
            return None

        # Calculate weights for temperature-weighted random choice (p_i ∝ (wt_i)^(1/T))
        weights = [ (wt ** (1.0 / self.temperature)) for _, wt in candidates ] # Apply temperature to weights for sampling
        total = sum(weights) # Sum all the calculated weights
        pick_point = self.rng.random() * total # Generate a random number within the total weight range
        acc = 0.0 # Initialize accumulator for weights
        for (uci_move, _), weight in zip(candidates, weights): # Iterate through candidates and their calculated weights
            # Accumulate weights until the pick point is reached
            acc += weight # Add current weight to accumulator
            if acc >= pick_point: # If accumulator reaches or exceeds the pick point
                chosen = legal_by_uci[uci_move] # Select the corresponding legal move
                # If underpromotion encoded (rare in openings), propagate it.
                if len(uci_move) == 5 and getattr(chosen, "is_pawn_promotion", False): # Check for underpromotion (UCI move string length 5) and if the chosen move is a pawn promotion
                    setattr(chosen, "promotion_choice", uci_move[4].upper()) # Set the promotion choice on the move object (e.g., 'q', 'r', 'b', 'n')
                return chosen # Return the chosen book move

        # Fallback: This case should ideally not be reached if total > 0
        return legal_by_uci[candidates[0][0]]
    
    
    """
    Debug helper functions
    """
    def peek(self, game_state) -> Tuple[Tuple[str, ...], Dict[str, int]]:
        # Return (current_prefix, candidate_map) without filtering by legality.
        prefix = _moves_log_to_uci(game_state.moves_log) # Convert game history to UCI prefix
        return prefix, self._prefix.get(prefix, {}) # Return prefix and its associated candidate moves

    def stats(self) -> Tuple[int, int]:
        return len(self._prefix), sum(len(v) for v in self._prefix.values()) # Count total prefixes and count of available next moves for each prefix.
    

    """
   Internal functions for building the opening book
    """
    # Builds the opening book from a Hugging Face dataset.
    def _build_from_hf(self, name: str, split: str, limit: Optional[int]) -> None:

        dataset = load_dataset(name, split=split) # Load the dataset from Hugging Face

        drop = ["img"] # Columns to drop from the dataset
        dataset = dataset.remove_columns(drop) # Remove the 'img' column to avoid unnecessary decoding
        entries_used = 0 # Counter for the number of dataset entries processed
        prefix: Dict[Tuple[str, ...], Dict[str, int]] = {} # Dictionary to store the opening book prefixes

        for row in dataset: # Iterate through each row (game) in the dataset
            uci_str = (row.get("uci") or "").strip() # Get the UCI move string for the current game
            if not uci_str: # Skip if the UCI string is empty
                continue
            seq = tuple(tok for tok in uci_str.split() if tok) # Split the UCI string into individual move tokens
            lim = min(len(seq), self.max_book_plies) # Determine the limit for plies to index from this game
            for i in range(lim): # Iterate up to the defined ply limit
                pre = seq[:i] # The current move sequence as a prefix
                nxt = seq[i] # The next move in the sequence
                bucket = prefix.setdefault(pre, {}) # Get or create the bucket for the current prefix
                bucket[nxt] = bucket.get(nxt, 0) + 1 # Increment the weight for the next move in the bucket

            entries_used += 1 # Increment the count of processed entries
            if limit is not None and entries_used >= limit: # Check if the processing limit has been reached
                break # Exit the loop if the limit is reached

        self._prefix = prefix # Assign the built prefix dictionary to the object's _prefix attribute

    
    """
    Cache load and save functions
    """
    # Load the opening book from cache file.
    def _try_load_cache(self, path: str) -> bool: 
        try:
            with gzip.open(path, "rb") as f: # Open the gzipped file in binary read mode
                obj = pickle.load(f) # Load the pickled object from the file
             # If the loaded object is a dictionary, assign it to _prefix and return True.
            if isinstance(obj, dict): # Check if the loaded object is a dictionary
                self._prefix = obj # Assign the loaded dictionary to the _prefix attribute
                return True # Indicate successful loading
        except Exception: # Catch any exceptions during loading (file not found, corrupted file etc)
            pass
        return False # Loading failed

    # Save the current opening book (_prefix) to the cache file.
    def _try_save_cache(self, path: str) -> None: #
        try:
            with gzip.open(path, "wb") as f: # Open the gzipped (cache) file in binary write mode
                pickle.dump(self._prefix, f, protocol=pickle.HIGHEST_PROTOCOL) # Store (Pickle) and dump the _prefix dictionary to the file
        except Exception: # Catch any exceptions during saving
            pass # Ignore the exception and proceed