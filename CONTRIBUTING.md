## Contributing

We welcome fixes, speedups, and docs!

1) **Fork** the repo and create a **branch** (`feature/...` or `fix/...`).
2) **Set up**: Follow the steps in the Setup section in the [README](/README.md) file/
3) **Quick checks** (if you touch search/eval) - using **[arena.ipynb](/arena.ipynb)**:
   - Run **perft** (we validate up to depth 5).
   - Do a short **arena** smoke test and note basic metrics (e.g., ai_nps, tt_hits, β-cutoffs).
4) **Commit & PR**:
   - Use clear commit messages.
   - Open a PR with a 2–3 line summary. If perf-related, include a tiny before/after table.

**Style:** clear Python, small functions, brief docstrings for non-obvious logic.  
**Env note:** developed/tested on TensorFlow 2.19.0.
