# ChessGPT

A lightweight chess bot that runs entirely on a laptop CPU. An **encoder-only transformer** (0.84M parameters) evaluates positions, and an **alpha-beta search** uses those evaluations to pick moves. You play in the terminal: type your move, and the bot answers in standard chess notation.

- **Tiny model:** 838,401 parameters. The checkpoint is 9.6 MB (the weights alone are about 3.4 MB).
- **CPU only:** trained and run without a GPU.
- **Distilled from Stockfish:** the network imitates Stockfish depth-10 evaluations from the Gigafish dataset (3.8B positions; we use a random 1M-position sample).
- **Rules from `python-chess`:** legal moves, castling, en passant, promotion, checkmate, stalemate, insufficient material, threefold repetition and the 50-move rule are handled by the library, not re-implemented.

## Results

**Validation metrics** (20,000 held-out positions; model trained on 1,000,000 positions for 3 epochs):

| Metric | Value | What it means |
|---|---|---|
| MSE | 0.134 | Mean squared error between predicted and Stockfish value, in the model's `tanh` space (-1 to 1) |
| MAE (centipawns) | 280.7 cp | Mean absolute error after converting values back to centipawns |
| Sign agreement | 89.63% | How often the model picks the right side as winning, on positions where Stockfish's eval is beyond ±100 cp |

**Play test (one game, so treat it as anecdotal):** the bot played against a 1500-rated chess.com player (the author) for 20 moves each.
- For about 7-9 moves it played almost perfectly, and it kept playing well for 1-2 moves after leaving its opening book.
- It was on a disadvantage of +7 against the human, but after the human switched to aggressive lines, it held its own and went down to only about 1.6 pawns worse, and then it blundered into a mate in one.

## Architecture

### Overview

```
 Gigafish (HF dataset: FEN + Stockfish d10 eval)
          │
          ▼
 ┌─────────────────┐   .npy shards    ┌─────────────┐   best.pt   ┌────────────┐
 │ data_pipeline.py│ ───────────────► │  model.py   │ ──────────► │ Evaluator  │
 │  encode + build │                  │ train + eval│             └─────┬──────┘
 └─────────────────┘                  └─────────────┘                   │ value in (-1,1)
          ▲ encode_board (shared)                                       ▼
                                                              ┌─────────────────┐
   you ◄──── SAN moves ────► play.py ◄───── best move ─────── │    search.py    │
                                                              │ alpha-beta + TT │
                                                              └─────────────────┘
                                                                python-chess
                                                              (legal moves, rules)
```

`encode_board` lives in `data_pipeline.py` and is imported by the model and the search. Training and play therefore use identical encoding.

### 1. Position encoding (`data_pipeline.py`)

Every position is encoded **from the side-to-move's perspective**. If Black is to move, the board is flipped vertically and Black's pieces are treated as "own". The model never sees absolute colors, which halves what it has to learn.

| Part | Encoding |
|---|---|
| Squares | 64 tokens; index = `8*rank + file`, with rank 0 = the side-to-move's back rank |
| Piece vocabulary (13) | `0` empty · `1-6` own P,N,B,R,Q,K · `7-12` opponent P,N,B,R,Q,K |
| Aux (0-143) | `castling_rights(0-15) * 9 + en_passant_code(0-8)` |
| Target | `tanh(clip(cp, ±2000) / 400)`, from the side-to-move's view |

The encoding deliberately leaves out the halfmove clock and move number. The 50-move rule and repetition are handled by the search. Mates and draws are scored exactly by the search, so the model only has to judge quiet positions.

**Dataset build:**
- The build picks up to 64 random parquet shards from the Hugging Face repo and reads a block from a random offset in each.
- It keeps every 10th row from each block, so a single game contributes few positions. It then shuffles, deduplicates on the first 4 FEN fields, and splits into train and validation by hashing the position.
- If the repo has no parquet shards, it falls back to buffered streaming (which is only locally shuffled).
- `inspect` determines whether the dataset's evals are from White's or the side-to-move's view (by correlating eval with material on Black-to-move positions), and `build` converts accordingly. The parser handles evals in pawns or centipawns and mate strings such as `#+3`.

### 2. Model (`model.py`)

```
tokens [64] ─► piece_emb(13,128) ─┐
square idx ─► pos_emb(64,128)  ───┴─ add ─► [64,128] ─┐
aux [0-143] ─► aux_emb(144,128) + CLS ───► [1,128] ───┴─► concat ─► [65,128]
                                                           │
                       4 × pre-LN encoder block:           ▼
                       LN → MultiHeadAttention(4 heads) → +residual
                       LN → FFN(128 → 512 → 128, GELU)  → +residual
                                                           │
                       take index 0 (global token) ◄───────┘
                       LN → Linear(128,128) → GELU → Linear(128,1) → tanh
                                                           │
                                         value v in (-1, 1), side-to-move view
```

| Setting | Value |
|---|---|
| d_model / layers / heads / d_ff | 128 / 4 / 4 / 512 |
| Sequence length | 65 (1 global token + 64 squares) |
| Parameters | 838,401 |
| Dropout | 0.0 |
| Optional | `--attn-bias`: learned per-head 65×65 attention bias (off by default) |

**Training:**
- AdamW, learning rate 1e-3, weight decay 0.01.
- 1,000 warmup steps, then cosine decay to 1e-5.
- Batch size 256, gradient clipping at 1.0, MSE loss.
- Validation and a checkpoint every 2,000 steps, with `last.pt` and `best.pt` (lowest validation MSE).
- The training run can be resumed with `--resume`.

The `Evaluator` class loads a checkpoint and scores batches of encoded positions with `evaluate_encoded` or `evaluate`.

### 3. Search (`search.py`)

Negamax alpha-beta with iterative deepening. Defaults: depth 4, 15 seconds per move.

| Component | Detail |
|---|---|
| Terminal scoring | Checkmate = `-(10 - 0.01·ply)`, so faster mates score higher. Stalemate, insufficient material, 50-move and repetition = 0. |
| Repetition | A position that has occurred once already counts as a draw inside the search. |
| Transposition table | Zobrist-keyed, up to 2M entries. Mate scores are not stored; only the best move is kept. |
| Evaluation cache | Zobrist-keyed, up to 1M entries. |
| Depth-1 batching | All children of a depth-1 node are evaluated in a single forward pass. |
| Move ordering | TT/PV move, then captures and promotions by MVV-LVA, then killer moves, then history heuristic. |
| Quiescence search | On by default: captures and promotions only (all legal moves when in check), capped at 4 plies. Disable with `--no-qsearch`. |
| Time control | Checked every 256 nodes. An unfinished iteration is discarded and the last completed depth is used. |
| Opening book | Encyclopedia of Chess Openings lines via the `Openix` library, plus an optional Polyglot `.bin` at `data/performance.bin`. A random book move is played while the position is in the book; then the search takes over. |

### 4. Game loop (`play.py`)

Reads your move, parses SAN or UCI, asks the searcher for a reply, and prints it in SAN. After every move it checks the game outcome with `claim_draw=True`, so the game ends automatically on checkmate, stalemate, insufficient material, threefold repetition and the 50-move rule. A PGN is printed at the end.

## Project structure

```
chessbot/
├── data_pipeline.py   # FEN encoding, dataset build, data loaders
├── model.py           # transformer, training loop, Evaluator
├── search.py          # alpha-beta search, opening book, MaterialEvaluator
├── play.py            # command-line game loop
├── requirements.txt
├── data/              # generated: .npy shards, meta.json, opening-book cache
└── checkpoints/       # generated: best.pt, last.pt
```

## Installation

Python 3.10 or newer.

```bash
pip install -r requirements.txt
# CPU-only PyTorch (smaller download), if you prefer:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## How to play

You only need `checkpoints/best.pt`. Run:

```bash
python play.py --color W        # you play White; the bot plays Black
python play.py --color B        # you play Black; the bot moves first
python play.py                  # asks which color you want
```

Note that `--color` is **your** color, not the bot's.

**The intended workflow:** the bot doesn't draw a board for you. Play on any board you like (chess.com, Lichess, or a physical set). Type your moves into the terminal, and the bot prints its reply in chess notation. Then make the bot's move on your board.

```
Your move: e4
Bot plays: e5
  depth=book eval=+0.00 nodes=1 time=0.0s
Your move: Nf3
Bot plays: Nc6
  depth=book eval=+0.00 nodes=1 time=0.0s
Your move: Bb5
Bot plays: a6
  depth=4 eval=+0.21 nodes=2840 time=6.3s
```

(Output above is illustrative.) The info line shows search depth (or `book`), the evaluation **from White's perspective in pawns** (`M+3` / `M-2` means mate for White / Black in 3 / 2), nodes searched, and time taken.

**Accepted move formats:**

| Format | Examples |
|---|---|
| Standard notation (SAN) | `e4`, `Nf3`, `Nxc6`, `exd5`, `Nbd7`, `O-O`, `O-O-O`, `e8=Q` |
| Zero-style castling | `0-0`, `0-0-0` |
| UCI | `e2e4`, `e7e8q` |

- Input is case-sensitive for pieces and files: `Bxc3` is a bishop capture, `bxc3` is a pawn capture.
- Trailing `+`, `#`, `!` and `?` are ignored.
- If you promote without naming a piece (`e8`), the bot asks `Promote to (Q/R/B/N)?`.
- Illegal, ambiguous or unreadable input prints an error and asks again.

**Commands** (type at the `Your move:` prompt):

| Command | Action |
|---|---|
| `help` | Show the command list |
| `moves` | List all legal moves in SAN |
| `board` | Print the board, oriented from your side (it is never printed automatically) |
| `fen` | Print the current position as FEN |
| `eval` | Show the model's evaluation of the current position (side-to-move view) |
| `undo` | Take back one move pair (refuses if fewer than 2 moves have been played) |
| `resign` | Resign; prints the PGN |
| `quit` | Exit |

**Options:**

| Flag | Default | Purpose |
|---|---|---|
| `--color W/B` | ask | Your color |
| `--checkpoint PATH` | `checkpoints/best.pt` | Model weights |
| `--depth N` | 4 | Maximum search depth |
| `--time SECONDS` | 15 | Time limit per move |
| `--fen "<FEN>"` | start position | Start from a custom position |
| `--no-qsearch` | off | Disable quiescence search |
| `--material` | off | Use a simple material evaluator instead of the model (no checkpoint needed) |
| `--threads N` | auto | PyTorch CPU threads |

Lower `--time` or `--depth` for faster replies, and raise them for stronger (slower) play.

## Training your own model

```bash
# 1. Inspect the dataset: detects columns, eval format and eval perspective, writes data/meta.json
python data_pipeline.py inspect --hf-dataset <huggingface_repo_id>

# 2. Build the dataset (random shards, encoded to .npy). Use --n-train 300000 for a quick pilot.
python data_pipeline.py build --hf-dataset <huggingface_repo_id> \
    --n-train 1000000 --n-val 20000 --out data --seed 42 --stride 10

# 3. Measure speed and estimate training time
python model.py bench --data data

# 4. Train (writes checkpoints/best.pt and last.pt)
python model.py train --data data --epochs 3 --batch-size 256
```

Training prints a train-vs-validation summary (MSE, MAE in centipawns, sign agreement) at the end of every epoch.

## Tests

Every file has a self-test:

```bash
python data_pipeline.py selftest      # encoding correctness, mirror invariance, value conversion
python model.py selftest --data data  # shapes, overfit gate, mirror symmetry, checkpoint round-trip
python search.py selftest             # mate-in-1/2, alpha-beta vs minimax, draws, time limit
python play.py selftest               # move parsing, castling, en passant, repetition, 50-move, mate
```

## Limitations

- **Tactical blunders:** the bot can still overlook short tactics. In the test game it blundered into a mate in one. The cause hasn't been investigated yet.
- **Shallow search:** a transformer evaluator on a CPU is slow, so the default is depth 4. Deeper search needs a faster evaluator, such as a smaller or quantized model.
- **Validation numbers measure imitation, not playing strength.** They compare the model to Stockfish depth-10 labels. The train/validation split is by position, not by game.
- **Anecdotal strength estimate:** the single game above is not a rating. A fair estimate would need many games against a range of opponents.
- **Opening book:** `play.py` has no flag to turn it off, and book moves are chosen at random, so games can differ between runs.

## Ideas for next steps

- Self-play matches (new vs. old checkpoint, with and without quiescence) to measure each change.
- int8 quantization or ONNX Runtime for faster evaluation, which would allow deeper search.
- Learned attention bias (`--attn-bias`) and a larger model or more training data.
- Investigate the blunder in the test game, and add a Stockfish-based game review for debugging.
