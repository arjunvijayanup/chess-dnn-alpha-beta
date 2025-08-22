<p align="center">
  <img src="assets/Logo.png" width="200" alt="Chess AI Logo">
</p>

<h1 align="center">Chess AI</h1>
<h2 align="center"> Negamax α–β Search with Incrementally Updatable DNN Evaluation</h2>

---

<p align="center">
  <img src="https://img.shields.io/badge/Python-≥3.10-blue?logo=python" />
  <img src="https://img.shields.io/badge/pygame-Latest-blue" />
  <img src="https://img.shields.io/badge/python--chess-Latest-blue" />
  <img src="https://img.shields.io/badge/numpy-Latest-blue" />
  <img src="https://img.shields.io/badge/pandas-Latest-blue" />
  <img src="https://img.shields.io/badge/tensorflow-2.19.0-orange?logo=tensorflow" />
  <img src="https://img.shields.io/badge/tqdm-Latest-lightgrey" />
  <img src="https://img.shields.io/badge/matplotlib-Latest-blue" />
  <img src="https://img.shields.io/badge/datasets-HF-blue" />
  <img src="https://img.shields.io/badge/Stockfish-Integrated-green" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
  <img src="https://img.shields.io/badge/Platform-macOS%20|%20Windows-lightgrey" />
</p> 
&#x20; &#x20;

<p align="justify">
This project implements a <b>hybrid</b> chess engine that integrates classical search techniques with a neural network–based evaluation function. The engine applies a <b>Negamax search with α–β pruning</b>, a standard approach in modern chess engines, to efficiently traverse the game tree. Board evaluation is performed by a <b>deep neural network (DNN)</b>, trained on encoded chess positions, which replaces traditional handcrafted evaluation rules. This combination allows the engine to capture tactical and positional patterns beyond those represented in simple numeric heuristics.
</p>

<p align="justify">
The system includes several supporting features: a transposition table to avoid redundant calculations, killer move and history heuristics to improve move ordering, and an opening book derived from Lichess data to strengthen early play. A Pygame-based graphical user interface (GUI) enables play in multiple configurations, including <b>Human vs AI, AI vs AI, and AI vs Stockfish</b>. Integration with the Stockfish engine provides a benchmark for performance comparisons. In fixed-depth arena matches, the engine achieved an estimated <b>Elo rating in the 1500–1800</b> range at depths 2–4.
</p>

<p align="justify">
Overall, the project demonstrates the integration of search, machine learning, and user interface design within a chess engine, highlighting both the potential and current limitations of Python-based AI systems.
</p>

---

## Table of Contents

- [Motivation](#motivation)
- [Highlights](#highlights)
- [Repository Layout](#repository-layout)
- [Setup](#setup)
  - [Windows](#windows)
  - [macOS](#macos)
  - [Run the UI](#run-the-ui)
  - [Quick Start Checklist](#quick-start-checklist)
- [Methodology](#methodology)
  - [Overview](#overview)  
  - [How the Engine Works](#how-the-engine-works)
    - [Game state & legal move generation](#game-state-legal-move)
    - [Search (Negamax + Alpha-Beta Pruning)](#search-negamax-ab-pruning)
    - [Repetition & “Ping-Pong” handling](#repetition-ping-pong-handling)
    - [Evaluation](#evaluation-tf-model)
    - [Move ordering heuristics](#move-ordering-heuristics)
    - [Incremental encodings & mini-batching](#incremental-encodings-mini-batching)
    - [Opening book](#opening-book)
    - [Stockfish integration](#stockfish-integration)
    - [Multiprocessing](#multiprocessing)
    - [UI/UX (Pygame)](#ui-ux-pygame)
    - [Score Table](#score-table)
  - [Network Architecture & Training](#network-architecture--training)
  - [Prediction Speedups](#prediction-speedups)
- [Benchmarking Against Stockfish](#benchmarking-against-stockfish)
  - [Observations](#observations)
- [Advantages & Limitations](#advantages--limitations)
- [Ablations, Lessons Learned](#ablations-lessons-learned)
- [Future Scope](#future-scope)
- [Poster](#poster)
- [License](#license)
- [References](#references)

---

## Motivation

The design of this engine is guided by two complementary inspirations in modern computer chess. The α–β search framework is motivated by Stockfish, which demonstrates the effectiveness of deep, optimized search with strong pruning techniques. For position evaluation, our approach draws inspiration from AlphaZero, which employs a full neural network model (in their case, a convolutional architecture). While we initially considered CNN-based evaluation, this was set aside due to practical constraints, and we instead adopted a fully connected DNN.

Implementing a full-model DNN in Python introduced its own efficiency challenges, particularly with inference speed during deep search. As a result, part of our motivation also shifted toward engineering improvements that reduce per-move computation time. The limitations of the initially considered CNN approach are discussed further in the Ablations section.

## Highlights

- **Negamax α–β** implementation with a lightweight **transposition table** and per‑search **stats**.
- **Neural evaluation** in TensorFlow/Keras using a **782‑dimensional** position encoding.
- **Incremental feature updates**: reuse parent features and update only what changed after a move.
- **Batch inference** with a cached prediction graph for minimal overhead inside the search loop.
- **Transposition table (TT)**: per-position cache to prevent redundant evaluation of positions and aid pruning.
- **Move ordering**: promotion move first, then captures, killer moves, and history‑boosted quiets.
- **Draw detection** implemented in the engine: threefold repetition, fifty‑move rule, insufficient material.
- **Pygame UI** to play Human/AI/Stockfish; **Stockfish** wired via `python-chess`.

---

## Repository Layout

```
project-ai-chess-engine/
├── chessAI.py                          # Search (negamax+αβ), TT, batching, heuristics, TF model IO
├── chessEngine.py                      # GameState, legal moves, repetition/fifty‑move/insufficient, FEN helpers
├── chessEncoding.py                    # 782‑dim input encoding + incremental update helpers (INPUT_DIM = 782)
├── chessOpening.py                     # OpeningBook using Lichess HF dataset; gzip+pkl cache
├── chessMain.py                        # Pygame UI and game loop (Human/AI/Stockfish; promotion & undo handlers)
├── stockfishHandler.py                 # OS‑aware launcher; move conversion via python‑chess
├── dnn_train/
│   └── ai_train.py                     # Training script (Dense MLP + Dropout; loss=mse, metric=mae)
│   └── launch_train_sagemaker.ipynb    # Training job launcher notebok
├── stats/                              # CSVs & notebooks with arena results, trained DNN metrics and summaries
│   ├── summary.csv, arena_results_*.csv, lichess_eval_model.csv, plots.ipynb
├── images_pieces/                      # Chess Piece images used by the UI
├── stockfish/                          # Stockfish executables (For Windows and Mac OS) 
│   └── stockfish-macos-m1-apple-silicon
│   └── stockfish-windows-x86-64-avx2.exe
├── arena.ipynb                         # Game simulation notebook for running/evaluating matches
├── lichess_eval_model.keras            # Trained DNN Model (Keras) evaluator
├── assets/                             # README media assets 
├── Poster.pdf                          # Poster
└── LICENSE.md                          # MIT License
```

---

## Setup

### ![Windows](https://img.shields.io/badge/Windows-0078D6?logo=windows&logoColor=white) 

```bash
python -m venv chess_env
chess_env\Scripts\activate.bat    # For powershell -> chess_env\Scripts\activate.ps1
pip install --upgrade pip
pip install pygame python-chess numpy pandas tensorflow==2.19.0 tqdm matplotlib datasets
```

### ![macOS](https://img.shields.io/badge/macOS-000000?logo=apple&logoColor=white)

```bash
python -m venv chess_env
source chess_env/bin/activate 
pip install --upgrade pip
pip install pygame python-chess numpy pandas tensorflow==2.19.0 tqdm matplotlib datasets
```

### Run the UI

```bash
python chessMain.py
```

In the Pygame start overlay, choose **White** and **Black** controllers (Human / AI / Stockfish) and their depths.

- **Move input**: click source → destination squares.  
- **Promotion**: ♕ <kbd>q</kbd> / ♖ <kbd>r</kbd> / ♗ <kbd>b</kbd> / ♘ <kbd>n</kbd>  
- **Undo**: ↩️ <kbd>z</kbd>  (reset: <kbd>r</kbd>)  

| ![](assets/chess_UI.gif) | ![](assets/human_vs_AI.gif) |
|:--:|:--:|
| *UI Startup* | *Human (white) vs Our AI (black)* |
| ![](assets/ai_vs_stockfish.gif) | ![](assets/human_vs_human.gif) |
|||
| *Our AI (white) vs Stockfish (black)* | *Human (white) vs Human (black)* |


### Quick Start Checklist
- **Stockfish binary**: Ensure the executable exists in `stockfish/` with the expected filename and execute permission.
- **TensorFlow compatibility:** Developed and tested on **TensorFlow 2.19.0**. Other versions may work, but are **not guaranteed**. Check your local version with:
  `python -c "import tensorflow as tf; print(tf.__version__)"`
- **Per-move time:** Hardware-dependent (CPU/GPU, drivers). If you experience lag, lower the search depth for smoother play.
- **First-run startup delay:** On first launch, the opening book is streamed and cached (from Lichess), which can add a short delay. Subsequent runs use the local cache and start quickly.

---

## Methodology

### Overview

Positions are encoded as a 782-D, white-perspective, NNUE-style feature vector and scored by a compact DNN. Search uses negamax + ${\alpha}-{\beta}$ pruning with per-parent minibatching at the leaf frontier (typically $\le$ 16 siblings in one NN call) to keep move ordering and pruning effective while reducing evaluator overhead. The engine is implemented in Python + TensorFlow and inference runs in float32 using a pre-compiled `tf.function` graph to minimize per-call Python/Keras overhead.

<p align="center">
  <img src="assets/Model Diag_final.png" alt="WorkingDiagram" width="800"><br>
  <em>Full system pipeline, including data encoding, the network architecture, batched search, and
the training environment.</em>
</p>

### How the Engine Works

<a id="game-state-legal-move"></a>
#### 1) Game state & legal move generation — **[`chessEngine.py`](./chessEngine.py)**

- `GameState` stores the board (8×8, two‑char piece codes like `wQ`, `bN`), side‑to‑move, castling rights, en‑passant, half‑move counter, and move log.
- **100% legal move generation** with per-piece helpers (`get_pawn_moves`, `get_rook_moves`, `get_knight_moves`, `get_bishop_moves`, `get_queen_moves`, `get_king_moves`) plus castling, en-passant and promotions.
- **Draw detection**: threefold repetition (via a repetition counter), fifty‑move rule, and insufficient material checks.
- **FEN I/O** utilities (`to_fen`, `from_fen`) for interoperability with external engines and tooling.

<a id="search-negamax-ab-pruning"></a>
#### 2) Search (negamax + $\alpha\beta$ pruning) — **[`chessAI.py`](./chessAI.py)**

- **Core**: depth-limited negamax with $\alpha\beta$ pruning (`negamax_alpha_beta_search(...)`,`get_best_move(...)`).
- **Transposition table (TT)**: a Python dict keyed by FEN with flags `EXACT`, `LOWER`, `UPPER` to bound or commit scores.
- **Move ordering**: see 5; ordering is applied before recursion each ply.

<a id="repetition-ping-pong-handling"></a>
#### 3) Repetition & “Ping-Pong” handling — **[`chessAI.py`](./chessAI.py)**

-  **Early draw detection**: Treat threefold repetition, 50-move rule, and insufficient material as terminal draws (searched no further).
-  **Ping-pong penalty**: When the position is drawish (|score| ≤ $\epsilon$), penalize immediate ABAB repetitions that don’t change material/rights (no capture/castle/promo). A small `PING_PONG_PENALTY` is applied only at the frontier and never cached as an EXACT TT score.

<a id="evaluation-tf-model"></a>
#### 4) Evaluation — **[`chessEncoding.py`](./chessEncoding.py)** + TF Model (`lichess_eval_model.keras`)

-  **Feature vector (782-D)**: 768 piece-plane encodings + 14 auxiliary features (side-to-move, castling rights, ep file, half-move clock, etc.).
-  **Network output**: white-centric score in [−1, 1] (blend-ready with other terms if needed).
-  **Model lifecycle**: the AI process loads the Keras model once (path set by `MODEL_PATH`). The first call compiles a `tf.function` prediction graph; later calls reuse it (lower overhead than model.predict in this setup).
-  **Terminal scoring**: checkmate ≫ draw ≫ worst evaluation; stalemate anchored at 0 to avoid horizon artifacts.

<a id="move-ordering-heuristics"></a>
#### 5) Move ordering heuristics — **[`chessAI.py`](./chessAI.py)**

-  **Priority**: Promotion > Capture > Killer > Quiet (history-ordered).
-  **Killer moves**: two per depth; updated on β-cutoff.
-  **History heuristic**: quiet-move history increments with a depth-squared term on β-cutoff; used as a tiebreaker within quiets.
-  This ordering increases early β-cutoffs and improves TT usefulness.

<a id="incremental-encodings-mini-batching"></a>
#### 6) Incremental encodings & mini-batching — **[`chessAI.py`](./chessAI.py)** + **[`chessEncoding.py`](./chessEncoding.py)**

-  **Incremental updates**: rather than recomputing the full 782-D vector, child positions **toggle only the changed indices** (move, capture, castling, ep, promotion).
-  **Frontier minibatching**: at the leaf frontier (`search_depth_left = 1`), **siblings are encoded together** and sent to the NN in a single batch (default batch ≈ 16) to amortize framework overhead without harming ordering.

<a id="opening-book"></a>
#### 7) Opening book — **[`chessOpening.py`](./chessOpening.py)**

-  **Data**: curated**[Lichess openings dataset](https://huggingface.co/datasets/Lichess/chess-openings)**.
-  **Usage**: temperature-sampled choices with a ply cap (≤20) to add variety while limiting early compute.
-  **Caching**: on first use, a pickled prefix map ( (e.g., `opening_moveCache.pkl.gz`) is created; subsequent runs read the cache for instant lookup.

| ![](assets/Indian_defense.png) | ![](assets/queens_gambit.png) | ![](assets/Zukertort_Opening.png) |
|:--:|:--:|:--:|
| *Indian Defense* | *Queens Gambit* | *Zukertort Opening* |

<a id="stockfish-integration"></a>
#### 8) Stockfish integration —  **[`stockfishHandler.py`](./stockfishHandler.py)**

-  **Handler**: `StockfishPlayer` encapsulates UCI I/O and converts between our `GameState` **(via FEN)** and `python-chess`.
-  **Binaries**: chooses an OS-specific default from the `stockfish/` folder; falls back to `shutil.which("stockfish")` if not executable.
-  **UI wiring**: the UI can run **separate instances per side** (e.g., human vs Stockfish, AI vs Stockfish) and currently exposes **fixed depth**; time controls and options (hash/threads/skill) are easy extensions.

<a id="multiprocessing"></a>
#### 9) Multiprocessing — **[`chessMain.py`](./chessMain.py)** + **[`chessAI.py`](./chessAI.py)**

-  The AI runs in a **separate process**; the UI enqueues requests and receives replies, keeping the Pygame loop responsive and **warming** the TF graph only once.

<a id="ui-ux-pygame"></a>
#### 10) UI/UX (Pygame) —  **[`chessMain.py`](./chessMain.py)**

-  Initializes Pygame, window and configuration.
-  Includes start menu to pick White/Black controllers (Human / our AI / Stockfish) and depths.
-  Calls our AI when it’s that side’s turn; launches Stockfish instances on demand.
-  Handles mouse and keys (Q/R/B/N promote, Z undo, R reset).
-  Orchestrates turns: sends/receives AI/Stockfish moves, show “thinking,” drop stale replies.
-  Updates GameState, log moves, draw and animate board, highlights and footer.
-  Detects checkmate/stalemate/draw and enable restart.
-  Cleanly shuts down AI/Stockfish on reset/exit.

###  Score Table

| Type                    | Name                  | Value               | 
|-------------------------|-----------------------|---------------------|
| Evaluation (terminal)   | Checkmate score       | ±1000               | 
| Evaluation (terminal)   | Stalemate score       | 0                   |
| Move ordering           | Promotion bonus       | +200                |
| Move ordering           | Capture bonus         | +100                |
| Move ordering           | Killer move bonus     | +50                 |
| Move ordering           | History score         | $depth^2$ (additive)|
| Draw handling           | Ping-pong penalty     | −0.02               |
| Draw handling           | Drawish threshold ($\epsilon$)   | 0.15     |                                              

---

###  Network Architecture & Training

[`ai_train.py`](dnn_train/ai_train.py) (TensorFlow/Keras):

| Section | Details |
|---|---|
| **Task/target** | Regression to normalized centipawn score (white-perspective) |
| **Model (DNN)** | `Dense(512, relu)` → `Dropout(0.2)` → `Dense(256, relu)` → `Dropout(0.2)` → `Dense(128, relu)` → `Dropout(0.2)` → `Dense(64, relu)` → `Dropout(0.2)` → `Dense(1, tanh)` |
| **Compile** | `loss="mse"`, `metrics=["mae"]` |
| **Optimizer** | Adam (`lr = 1e-4`) |
| **Data** | Streamed from [Lichess position evaluations](https://huggingface.co/datasets/Lichess/chess-position-evaluations) (384M rows); fixed validation set **N = 100k** |
| **Label scaling** | clip to ±1000; scale y= cp /1000 so **y∈[−1,1]); mates → ±1** |
| **Schedule** | **Epochs:** 20 <br> **Steps/Epoch:** 2000 <br> **Batch Size:** 1024 <br> **Callbacks:** ReduceLROnPlateau, EarlyStopping, ModelCheckpoint |
| **Compute & launch (AWS SageMaker)** | **Instance:** NVIDIA T4 (16 GB) <br> **Launch notebook:** [`launch_train_sagemaker.ipynb`](dnn_train/launch_train_sagemaker.ipynb) <br> **Total positions seen:** `20 × 2000 × 1024 = 40,960,000` ≈ **40.9M** <br> **Training time:** ~12 hours |

- **Observation:** Training loss improves faster than validation (expected due to noisy/streamed positions). Slow MSE convergence for validation data.
- **Export**: Save the best model as `.keras` and place it at repo root (e.g., `lichess_eval_model.keras`).

<p align="center">
  <img src="assets/lossplot.png" alt="Train/Validation loss" width="800"><br>
  <em>Training vs validation loss (MSE) over 20 epochs.</em>
</p>

---

### Prediction Speedups

-  **Compiled prediction graph (`tf.function`):** We wrap the trained model once and make a warm-up call to build a cached graph. Subsequent inferences reuse the cached graph (minimal Python overhead) and were faster than `model.predict()` in our setup.
  
  ```python
  # chessAI.py
  @tf.function(jit_compile=False)
  def build_prediction_graph(model):
      def predict_fn(x):
          return model(x, training=False)  # disables Dropout, uses moving averages
      return predict_fn

  prediction_graph = build_prediction_graph(eval_model)
  _ = prediction_graph(MODEL_INPUT)  # warm-up to compile once
  # ... later:
  preds = prediction_graph(batch_np_float32)  # used instead of model.predict(...)
  ```
___

---

## Benchmarking Against Stockfish

We ran **50 fixed-depth games for each (our-depth, Stockfish-depth) combination** to evaluate the performance of our AI. The plot and table below report **W–D–L outcomes** and **search-efficiency metrics**, respectively.

<p align="center">
  <img src="assets/stockfishvsAI.png" alt="stockfishvsAI" width="800"><br>
  <em>Arena Outcomes vs Stockfish — 50 game simulations per depth combination, W–D–L.</em>
</p>

<p align="center">
  <img src="assets/stockfishvsAI_Table.png" alt="stockfishvsAI_Table" width="800"><br>
  <em>Fixed-depth arena matches vs Stockfish (50 games each). Time metrics are medians;
NPS/TT/β-cutoffs/killer/history are pooled averages. Positive ΔElo favors our AI. TT = transposition table.</em>
</p>

Detailed results and additional analyses are available in `arena_results_*.csv` files in the [`stats/`](stats/) directory. **Also in the CSVs (not shown in the summary table):**
  - `tt_probes`, `tt_hit_rate`, `tt_stores`: Detailed TT activity
  - `first_move_cutoffs`: $\beta$-cutoffs caused by the first tried move
  - `avg_branch`: Mean branching factor
  - `ai_nodes`, `ai_moves`: Nodes searched and AI moves per game
  - `game_wall_ms`: Total wall time per game
  - `did_castle`, `did_en_passant`, `did_promotion`: Special-move flags
  - `result`: Game outcome

### Observations

- **Perft validation:** Depth **1–5** passed; all generated moves were **legal**.
- **Special moves:** Engine correctly executed **pawn promotions**, **castling** (both sides), and **en passant**.
- **Draw detection:** Observed all standard draws in arena play:
  - **Threefold repetition** (most common),
  - **Insufficient material**,
  - **50-move rule**.
- **Per-move time vs depth:** Median AI time per move **increased with depth** due to exponential node growth, higher effective branching, and more NN leaf evaluations (mini-batching ≤16 mitigates but does not remove this).
- **Transposition table (TT) usage vs depth:** As the search depth increased, the transposition table demonstrated a **higher hit rate**, reflecting **greater reuse of previously explored positions**.
- **Move-ordering/pruning stats vs depth:** Average **$\beta$-cutoffs**, **killer move uses**, and **history lookups** **increased with depth**, indicating effective ordering and pruning as node counts grow.
- **Rough strength estimate:** Using Stockfish at depths **2/3/4** as ~**1700/1770/1830 Elo** baselines, our engine’s estimated Elo is **~1500–1800** at depths **2–4**.
> ***NOTE**: Stockfish's elo at various depths were roughly estimated by referring the following publication: [D. R. Ferreira, “The Impact of Search Depth on Chess Playing Strength” (PDF)](https://web.ist.utl.pt/diogo.ferreira/papers/ferreira13impact.pdf)*
---

## Advantages & Limitations

### Advantages
- **Rules-correctness:** Fully legal move generation with robust handling of castling, en passant, and promotions.
- **Neural evaluation gains:** Stronger than classical numeric heuristics at comparable depth.
- **Search efficiency:** ${\alpha}–{\beta}$ pruning + move ordering (promotions/captures/killers/history) + mini-batched leaf NN evaluation + transposition table.
- **Fast inference path:** Compiled `tf.function` graph + one-time model load via multiprocessing.
- **Resilience features:** Early draw detection and bias away from aimless repetition in near-equal positions.
- **Opening book:** Adds variety and reduces early compute.

### Limitations
- **Throughput vs C++ engines:** Python implementation (loops, interpreter) is slower, as effective depth scales with CPU/GPU and RAM.
- **Depth comparability:** Raw ply counts aren’t like-for-like vs. Stockfish. Compare configs (level/options) or time controls instead.
- **Startup cost:** Model load introduces latency (partly mitigated by warm-start/compiled graph).
- **Shallow search errors:** Occasional blunders at low depth remain.

---

## Ablations, Lessons Learned

What we tried, measured, and rolled back—brief reasons why.

- **Move caching per-position (pre-computed; not TT):** Reduced repeated compute, but **startup/memory overhead** dominated in Python; net slowdown.
- **Null-move pruning:** Fewer recursive calls, yet **interpreter + function-call overhead** in Python outweighed gains as pruning effectiveness didn’t reduce overall runtime.
- **Quiescence search:** Conceptually sound (extend captures/tactics to reduce horizon effects), but in our pipeline **overhead > benefit**; NN leaf evaluation + move ordering was sufficient.
- **Extra hand-crafted terms (mobility, king safety) mixed into NN output:** Led to **double-counting** and unstable scales; simpler **NN-only score** proved more consistent.
- **Large NN batches:** Better throughput per call, but **hurt ${\alpha}–{\beta}$ move ordering** and delayed cutoffs; sweet spot remains **minibatch ≤ 16** at the frontier.
- **CNN evaluator (12×8×8 conv net):** Trained a small CNN, but **latency increased** with no clear accuracy gain at the same budget. On small batches, per-call overhead from **tensor conversion**, reshaping to (batch_size,12,8,8), host->device copy, and kernel launch dominates. The DNN delivered lower per-leaf latency.

---

## Future Scope

- **NNUE accumulator + INT8 head:** Cache the first-layer output in NN i.e. $(W_1 x + b_1)$ and update it per move (skip recomputing $W_1 x$\); run the remaining layers in **INT8** with SIMD (TFLite/ONNX) for big CPU speedups. Post-training quantization — **no retrain needed**.
- **Iterative deepening with time controls**: Practical play (per-move/total clocks); industry norm.
- **Reinforcement learning**: Useful but heavier infrastructure; add on top of supervised model.
- **Cython/C++ hotspots**: Port performance critical functions such as move-generation, make/undo, Transposition table probes, etc for further throughput.
- **Explore MCTS hybrid**: Prototype a policy/value Monte Carlo Tree Search (MCTS) variant (less classical than α–β) to compare strength vs compute trade-offs.
- **Zobrist hashing for TT (replace FEN):** 64-bit incremental key updated on make/undo; avoids FEN build/parse, cuts allocations, speeds probes—industry standard for fast position keys with near-zero duplicates.
- **Endgame module**: Add basic endgame heuristics for perfect small endings, faster mate/draw recognition and stronger practical play.

---

## Poster

The project poster provides a high-level overview — abstract, methodology, features, results, and future work.

**PDF:** [View the poster](Poster.pdf)

---
## License

This project is licensed under the MIT License. Please have a look at the [LICENSE](LICENSE) for more details.

---
## References

- [Eddie Sharick, “Creating a Chess Engine in Python”](https://youtube.com/playlist?list=PLBwF487qi8MGU81nDGaeNE1EnNEPYWKY_)
- [Klein, D. (2022). "Neural Networks for Chess."](https://doi.org/10.48550/arXiv.2209.01506)
- [Hugging Face, “Hugging Face — The AI community building the future.”](https://huggingface.co/)
- [Amazon Web Services, “Amazon SageMaker — Documentation.”](https://docs.aws.amazon.com/sagemaker/)
- [D. R. Ferreira, “The Impact of Search Depth on Chess Playing Strength” (PDF)](https://web.ist.utl.pt/diogo.ferreira/papers/ferreira13impact.pdf)
- [Stockfish Developers, “Stockfish” (GitHub repository)](https://github.com/official-stockfish/Stockfish)
- [Chessprogramming Wiki, “Chessprogramming.org”](https://www.chessprogramming.org/)
- [Lichess, “Lichess.org”](https://lichess.org/)
