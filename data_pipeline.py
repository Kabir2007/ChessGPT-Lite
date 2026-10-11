#!/usr/bin/env python3
"""File 1: FEN encoding, dataset build, data loaders.

CLI subcommands:
    inspect  – stream a few rows from the HuggingFace dataset, determine
               column names, eval format, and eval perspective (White-POV vs STM-POV).
    build    – download, sample, encode, and save train/val .npy arrays.
    selftest – run correctness checks on encoding and conversion functions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

import chess
import numpy as np
from tqdm import tqdm

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Constants (section 3.4)
# ---------------------------------------------------------------------------
K = 400
CP_CLIP = 2000


# ---------------------------------------------------------------------------
# Value conversion (section 3.4)
# ---------------------------------------------------------------------------
def cp_to_value(cp: float) -> float:
    """Convert centipawn score to tanh-space target in (-1, 1)."""
    return math.tanh(max(-CP_CLIP, min(CP_CLIP, cp)) / K)


def value_to_cp(v: float) -> float:
    """Convert tanh-space value back to centipawns."""
    v = max(-0.9999, min(0.9999, v))
    return K * math.atanh(v)


# ---------------------------------------------------------------------------
# Board encoding – fast path (section 3.2 – 3.3)
# ---------------------------------------------------------------------------
def encode_board(board: chess.Board) -> tuple[np.ndarray, int]:
    """Encode a board from the side-to-move's perspective.

    Returns
    -------
    tokens : np.ndarray, dtype uint8, shape (64,)
        Token vocabulary:
            0       = empty
            1 .. 6  = STM's own P, N, B, R, Q, K
            7 .. 12 = opponent's P, N, B, R, Q, K
        Index i = 8*rank + file where rank 0 is STM's back rank.
    aux : int in 0..143
        ``castling * 9 + ep_code`` (see spec section 3.3).
    """
    tokens = np.zeros(64, dtype=np.uint8)
    turn = board.turn  # chess.WHITE = True
    stm = turn
    opp = not turn

    # Flip mask: if Black to move, XOR square index with 56
    flip = 0 if turn else 56

    for piece_type in range(1, 7):  # PAWN=1 .. KING=6
        # STM pieces → tokens 1..6
        bb = board.pieces_mask(piece_type, stm)
        while bb:
            sq = (bb & -bb).bit_length() - 1
            tokens[sq ^ flip] = piece_type
            bb &= bb - 1

        # Opponent pieces → tokens 7..12
        bb = board.pieces_mask(piece_type, opp)
        while bb:
            sq = (bb & -bb).bit_length() - 1
            tokens[sq ^ flip] = piece_type + 6
            bb &= bb - 1

    # Aux value -----------------------------------------------------------
    castling = 0
    if board.has_kingside_castling_rights(stm):
        castling |= 1
    if board.has_queenside_castling_rights(stm):
        castling |= 2
    if board.has_kingside_castling_rights(opp):
        castling |= 4
    if board.has_queenside_castling_rights(opp):
        castling |= 8

    ep_code = 0
    if board.has_legal_en_passant():
        ep_code = 1 + chess.square_file(board.ep_square)

    aux = castling * 9 + ep_code
    return tokens, aux


# ---------------------------------------------------------------------------
# Board encoding – slow reference (section 4, testing only)
# ---------------------------------------------------------------------------
def encode_board_reference(board: chess.Board) -> tuple[np.ndarray, int]:
    """Reference encoder using piece_map() – slow, for testing only."""
    tokens = np.zeros(64, dtype=np.uint8)
    turn = board.turn
    flip = 0 if turn else 56

    for sq, piece in board.piece_map().items():
        mapped = sq ^ flip
        pt = piece.piece_type  # 1..6
        if piece.color == turn:
            tokens[mapped] = pt
        else:
            tokens[mapped] = pt + 6

    # Aux (identical logic)
    stm = turn
    opp = not turn
    castling = 0
    if board.has_kingside_castling_rights(stm):
        castling |= 1
    if board.has_queenside_castling_rights(stm):
        castling |= 2
    if board.has_kingside_castling_rights(opp):
        castling |= 4
    if board.has_queenside_castling_rights(opp):
        castling |= 8

    ep_code = 0
    if board.has_legal_en_passant():
        ep_code = 1 + chess.square_file(board.ep_square)

    aux = castling * 9 + ep_code
    return tokens, aux


# ---------------------------------------------------------------------------
# Eval parsing
# ---------------------------------------------------------------------------
def parse_eval_string(eval_str: str, unit: str = "auto") -> tuple[float | None, bool]:
    """Parse the eval column from the dataset.

    Returns (cp_value, is_mate).
    - If unit == "pawns", converts pawns to centipawns (* 100.0).
    - If unit == "centipawns" or "cp", returns value directly.
    - If unit == "auto", detects if value is in pawns (has decimal or magnitude <= 50.0).
    - Mate scores: ``#+N`` or ``#N`` → +CP_CLIP, ``#-N`` → -CP_CLIP.
    - Returns (None, False) on failure.
    """
    if eval_str is None:
        return None, False
    s = str(eval_str).strip()
    if not s:
        return None, False

    # Mate notation
    if s.startswith("#") or s.startswith("#+") or s.startswith("#-"):
        rest = s[1:]  # after '#'
        try:
            mate_moves = int(rest)
        except ValueError:
            return None, False
        return (float(CP_CLIP) if mate_moves >= 0 else float(-CP_CLIP)), True

    # Plain centipawn / pawn
    try:
        val = float(s)
        u = unit.lower()
        if u in ("pawns", "pawn"):
            val *= 100.0
        elif u == "auto":
            # If formatted as float with decimal or magnitude <= 50.0, it is in pawns (e.g. 0.6, -11.87)
            if abs(val) <= 50.0 and ("." in s or abs(val) < 20.0):
                val *= 100.0
        return val, False
    except ValueError:
        return None, False


# ---------------------------------------------------------------------------
# Inspect subcommand (section 5.1)
# ---------------------------------------------------------------------------
def cmd_inspect(args: argparse.Namespace) -> None:
    """Stream a few rows, determine column names, eval format, and POV."""
    from datasets import load_dataset

    repo_id = args.hf_dataset
    print(f"=== Inspecting dataset: {repo_id} ===\n")

    # Stream 20 rows
    ds = load_dataset(repo_id, split="train", streaming=True)
    rows = []
    for i, row in enumerate(ds):
        if i >= 20:
            break
        rows.append(row)
        if i == 0:
            print(f"Column names: {list(row.keys())}")
            print(f"Dtypes: { {k: type(v).__name__ for k, v in row.items()} }")
            print()

    # Determine FEN and eval column names
    fen_col = args.fen_col
    eval_col = args.eval_col
    if fen_col is None:
        for k in rows[0]:
            if "fen" in k.lower():
                fen_col = k
                break
    if eval_col is None:
        for k in rows[0]:
            if "eval" in k.lower():
                eval_col = k
                break

    if fen_col is None or eval_col is None:
        print(f"ERROR: Could not auto-detect columns. Available: {list(rows[0].keys())}")
        print("Specify --fen-col and --eval-col explicitly.")
        sys.exit(1)

    print(f"FEN column: '{fen_col}'")
    print(f"Eval column: '{eval_col}'")
    print()

    # Print sample rows
    print("--- First 5 rows ---")
    for i, row in enumerate(rows[:5]):
        print(f"  [{i}] FEN: {row[fen_col]}")
        print(f"       Eval: {row[eval_col]}")
    print()

    # Check eval format
    print("--- Eval format analysis ---")
    mate_count = 0
    cp_count = 0
    parse_fail = 0
    sample_values = []
    for row in rows:
        val_str = str(row[eval_col]).strip()
        if not val_str.startswith("#"):
            try:
                sample_values.append(float(val_str))
            except ValueError:
                pass

    eval_unit = "pawns" if sample_values and max(abs(v) for v in sample_values) <= 50.0 else "centipawns"
    if sample_values:
        print(f"  Detected eval unit: {eval_unit.upper()} (range: [{min(sample_values):.2f}, {max(sample_values):.2f}])")

    for row in rows:
        cp, is_mate = parse_eval_string(str(row[eval_col]), unit=eval_unit)
        if cp is None:
            parse_fail += 1
        elif is_mate:
            mate_count += 1
        else:
            cp_count += 1
    print(f"  CP values: {cp_count}, Mate values: {mate_count}, Parse failures: {parse_fail}")

    # Check FEN fields
    fen_sample = rows[0][fen_col]
    fen_parts = fen_sample.split()
    print(f"  FEN fields: {len(fen_parts)} ({'includes halfmove/fullmove' if len(fen_parts) >= 6 else 'missing halfmove/fullmove'})")
    print()

    # POV detection: sample 20,000 rows with Black to move
    print("--- POV detection (sampling 20,000 Black-to-move rows) ---")
    ds2 = load_dataset(repo_id, split="train", streaming=True)
    black_rows = []
    scanned = 0
    for row in ds2:
        scanned += 1
        fen = row[fen_col]
        parts = fen.split()
        if len(parts) >= 2 and parts[1] == "b":
            cp, is_mate = parse_eval_string(str(row[eval_col]), unit=eval_unit)
            if cp is not None and not is_mate:
                black_rows.append((fen, cp))
        if len(black_rows) >= 20000:
            break
        if scanned >= 200000:
            break

    print(f"  Collected {len(black_rows)} Black-to-move rows (scanned {scanned})")

    if len(black_rows) < 1000:
        print("  WARNING: Not enough Black-to-move rows for POV detection. STOP.")
        sys.exit(1)

    # Compute material balance from White's perspective
    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                    chess.ROOK: 5, chess.QUEEN: 9}
    materials = []
    evals = []
    for fen, cp in black_rows:
        try:
            b = chess.Board(fen)
        except ValueError:
            continue
        mat = 0
        for pt, val in piece_values.items():
            mat += len(b.pieces(pt, chess.WHITE)) * val
            mat -= len(b.pieces(pt, chess.BLACK)) * val
        materials.append(mat)
        evals.append(cp)

    materials = np.array(materials, dtype=np.float64)
    evals = np.array(evals, dtype=np.float64)

    # Pearson correlation
    if np.std(materials) < 1e-9 or np.std(evals) < 1e-9:
        print("  ERROR: Zero variance in materials or evals. Cannot compute correlation.")
        sys.exit(1)

    corr = np.corrcoef(materials, evals)[0, 1]
    print(f"  Pearson correlation (White-material vs eval): {corr:.4f}")

    if corr > 0.3:
        eval_pov = "white"
        print("  → Eval is WHITE-POV")
    elif corr < -0.3:
        eval_pov = "stm"
        print("  → Eval is STM-POV")
    else:
        print(f"  → INCONCLUSIVE (corr = {corr:.4f}). STOP.")
        sys.exit(1)

    # Save meta.json
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "eval_pov": eval_pov,
        "eval_unit": eval_unit,
        "fen_col": fen_col,
        "eval_col": eval_col,
        "mate_encoding": "#+N / #-N" if mate_count > 0 else "unknown",
        "fen_fields": len(fen_parts),
        "correlation": float(corr),
        "source_dataset": repo_id,
    }
    meta_path = out_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Saved {meta_path}")
    print("\n=== Inspect complete ===")


# ---------------------------------------------------------------------------
# Build subcommand (section 5.2)
# ---------------------------------------------------------------------------
def cmd_build(args: argparse.Namespace) -> None:
    """Sample, encode, and save train/val arrays."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files

    repo_id = args.hf_dataset
    n_train = args.n_train
    n_val = args.n_val
    out_dir = Path(args.out)
    seed = args.seed
    stride = args.stride

    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"

    # Load meta.json from inspect
    if not meta_path.exists():
        print("ERROR: data/meta.json not found. Run 'inspect' first.")
        sys.exit(1)
    with open(meta_path) as f:
        meta = json.load(f)

    fen_col = meta["fen_col"]
    eval_col = meta["eval_col"]
    eval_pov = meta["eval_pov"]
    eval_unit = meta.get("eval_unit", "auto")
    print(f"=== Building dataset from {repo_id} ===")
    print(f"  n_train={n_train}, n_val={n_val}, seed={seed}, stride={stride}")
    print(f"  eval_pov={eval_pov}, eval_unit={eval_unit}, fen_col={fen_col}, eval_col={eval_col}")
    print()

    rng = random.Random(seed)

    # Step 1: List repo files and find parquet shards
    print("Listing repo files...")
    all_files = list_repo_files(repo_id, repo_type="dataset")
    parquet_files = [f for f in all_files if f.endswith(".parquet")]
    print(f"  Found {len(parquet_files)} parquet files")

    if len(parquet_files) == 0:
        print("ERROR: No parquet files found. Trying streaming fallback...")
        _build_streaming_fallback(args, meta)
        return

    # Step 2: Pick files uniformly at random
    n_picked = min(64, len(parquet_files))
    picked_files = rng.sample(parquet_files, n_picked)
    print(f"  Picked {n_picked} files for sampling")

    # Step 3: From each file, read a contiguous block
    target_total = n_train + n_val
    # Multiply by stride because we only keep 1 out of every `stride` rows from the block
    rows_per_file = min(1_000_000, max(1, int(math.ceil(2.2 * target_total * stride / n_picked))))

    all_rows = []
    for i, pf in enumerate(tqdm(picked_files, desc="Downloading & reading shards")):
        try:
            local_path = hf_hub_download(repo_id, pf, repo_type="dataset")
            table = pq.read_table(local_path, columns=[fen_col, eval_col])
            n_rows_in_file = len(table)

            # Random start offset
            if n_rows_in_file <= rows_per_file:
                start = 0
                end = n_rows_in_file
            else:
                start = rng.randint(0, n_rows_in_file - rows_per_file)
                end = start + rows_per_file

            block = table.slice(start, end - start)

            # Stride: keep every Nth row
            indices = list(range(0, len(block), stride))
            for idx in indices:
                row_fen = block.column(fen_col)[idx].as_py()
                row_eval = block.column(eval_col)[idx].as_py()
                all_rows.append((row_fen, str(row_eval)))

        except Exception as e:
            print(f"  WARNING: Failed to read {pf}: {e}")
            continue

    print(f"\n  Collected {len(all_rows)} rows after stride")

    # Step 4: Shuffle and process
    rng.shuffle(all_rows)

    seen_keys = set()
    train_data = []
    val_data = []
    skip_parse = 0
    skip_eval = 0
    skip_no_moves = 0
    skip_dup = 0

    for fen, eval_str in tqdm(all_rows, desc="Processing rows"):
        if len(train_data) >= n_train and len(val_data) >= n_val:
            break

        # Parse FEN
        try:
            board = chess.Board(fen)
        except (ValueError, TypeError):
            skip_parse += 1
            continue

        # Skip if eval is null
        cp, is_mate = parse_eval_string(eval_str, unit=eval_unit)
        if cp is None:
            skip_eval += 1
            continue

        # Skip if no legal moves
        if not any(board.legal_moves):
            skip_no_moves += 1
            continue

        # Dedupe on first 4 FEN fields
        parts = fen.split()
        key = " ".join(parts[:4])
        if key in seen_keys:
            skip_dup += 1
            continue
        seen_keys.add(key)

        # Split: md5(key) % 50 == 0 → val, else train
        h = int(hashlib.md5(key.encode()).hexdigest(), 16)
        is_val = (h % 50 == 0)

        if is_val and len(val_data) >= n_val:
            is_val = False
        if not is_val and len(train_data) >= n_train:
            is_val = True

        if is_val and len(val_data) >= n_val:
            continue
        if not is_val and len(train_data) >= n_train:
            continue

        # Convert eval to STM perspective
        cp_stm = cp
        if eval_pov == "white" and board.turn == chess.BLACK:
            cp_stm = -cp

        # Encode
        tokens, aux = encode_board(board)
        target = cp_to_value(cp_stm)

        entry = (tokens, aux, target)
        if is_val:
            val_data.append(entry)
        else:
            train_data.append(entry)

    print(f"\n=== Build summary ===")
    print(f"  Train: {len(train_data)}")
    print(f"  Val:   {len(val_data)}")
    print(f"  Skipped - parse error: {skip_parse}")
    print(f"  Skipped - eval null:   {skip_eval}")
    print(f"  Skipped - no moves:    {skip_no_moves}")
    print(f"  Skipped - duplicate:   {skip_dup}")

    if len(train_data) < n_train or len(val_data) < n_val:
        print("  WARNING: Did not reach target counts. Consider increasing sample size.")

    # Save arrays
    def save_split(split_name, data):
        tokens_arr = np.array([d[0] for d in data], dtype=np.uint8)
        aux_arr = np.array([d[1] for d in data], dtype=np.uint8)
        targets_arr = np.array([d[2] for d in data], dtype=np.float32)

        np.save(out_dir / f"{split_name}_tokens.npy", tokens_arr)
        np.save(out_dir / f"{split_name}_aux.npy", aux_arr)
        np.save(out_dir / f"{split_name}_targets.npy", targets_arr)
        print(f"  Saved {split_name}: tokens{tokens_arr.shape}, aux{aux_arr.shape}, targets{targets_arr.shape}")

    save_split("train", train_data)
    save_split("val", val_data)

    # Target histogram
    all_targets = np.array([d[2] for d in train_data + val_data])
    hist, bin_edges = np.histogram(all_targets, bins=10)
    print("\n  Target histogram:")
    for i in range(len(hist)):
        print(f"    [{bin_edges[i]:+.3f}, {bin_edges[i+1]:+.3f}): {hist[i]}")

    # Update meta.json
    meta.update({
        "n_train": len(train_data),
        "n_val": len(val_data),
        "skip_parse": skip_parse,
        "skip_eval": skip_eval,
        "skip_no_moves": skip_no_moves,
        "skip_dup": skip_dup,
        "K": K,
        "CP_CLIP": CP_CLIP,
    })
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Updated {meta_path}")
    print("\n=== Build complete ===")


def _build_streaming_fallback(args: argparse.Namespace, meta: dict) -> None:
    """Fallback: streaming with buffer shuffle (section 5.2 step 5)."""
    from datasets import load_dataset

    repo_id = args.hf_dataset
    n_train = args.n_train
    n_val = args.n_val
    out_dir = Path(args.out)
    seed = args.seed
    stride = args.stride

    fen_col = meta["fen_col"]
    eval_col = meta["eval_col"]
    eval_pov = meta["eval_pov"]
    eval_unit = meta.get("eval_unit", "auto")

    print("WARNING: Falling back to streaming with buffer shuffle.")
    print("  The sample is only locally shuffled.")

    rng = random.Random(seed)
    skip_offset = rng.randint(0, 100_000)
    ds = load_dataset(repo_id, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=200_000)

    # Skip some rows for randomness
    it = iter(ds)
    for _ in range(skip_offset):
        try:
            next(it)
        except StopIteration:
            break

    seen_keys = set()
    train_data = []
    val_data = []
    skip_parse = 0
    skip_eval = 0
    skip_no_moves = 0
    skip_dup = 0
    row_count = 0

    target_total = n_train + n_val

    for row in tqdm(it, desc="Streaming rows", total=2 * target_total):
        if len(train_data) >= n_train and len(val_data) >= n_val:
            break

        row_count += 1
        if row_count % stride != 0:
            continue

        fen = row[fen_col]
        eval_str = str(row[eval_col])

        try:
            board = chess.Board(fen)
        except (ValueError, TypeError):
            skip_parse += 1
            continue

        cp, is_mate = parse_eval_string(eval_str, unit=eval_unit)
        if cp is None:
            skip_eval += 1
            continue

        if not any(board.legal_moves):
            skip_no_moves += 1
            continue

        parts = fen.split()
        key = " ".join(parts[:4])
        if key in seen_keys:
            skip_dup += 1
            continue
        seen_keys.add(key)

        h = int(hashlib.md5(key.encode()).hexdigest(), 16)
        is_val = (h % 50 == 0)

        if is_val and len(val_data) >= n_val:
            is_val = False
        if not is_val and len(train_data) >= n_train:
            is_val = True

        if is_val and len(val_data) >= n_val:
            continue
        if not is_val and len(train_data) >= n_train:
            continue

        cp_stm = cp
        if eval_pov == "white" and board.turn == chess.BLACK:
            cp_stm = -cp

        tokens, aux = encode_board(board)
        target = cp_to_value(cp_stm)

        entry = (tokens, aux, target)
        if is_val:
            val_data.append(entry)
        else:
            train_data.append(entry)

    print(f"\n=== Build summary (streaming) ===")
    print(f"  Train: {len(train_data)}")
    print(f"  Val:   {len(val_data)}")
    print(f"  Skipped - parse error: {skip_parse}")
    print(f"  Skipped - eval null:   {skip_eval}")
    print(f"  Skipped - no moves:    {skip_no_moves}")
    print(f"  Skipped - duplicate:   {skip_dup}")

    def save_split(split_name, data):
        tokens_arr = np.array([d[0] for d in data], dtype=np.uint8)
        aux_arr = np.array([d[1] for d in data], dtype=np.uint8)
        targets_arr = np.array([d[2] for d in data], dtype=np.float32)

        np.save(out_dir / f"{split_name}_tokens.npy", tokens_arr)
        np.save(out_dir / f"{split_name}_aux.npy", aux_arr)
        np.save(out_dir / f"{split_name}_targets.npy", targets_arr)
        print(f"  Saved {split_name}: tokens{tokens_arr.shape}, aux{aux_arr.shape}, targets{targets_arr.shape}")

    save_split("train", train_data)
    save_split("val", val_data)

    all_targets = np.array([d[2] for d in train_data + val_data])
    hist, bin_edges = np.histogram(all_targets, bins=10)
    print("\n  Target histogram:")
    for i in range(len(hist)):
        print(f"    [{bin_edges[i]:+.3f}, {bin_edges[i+1]:+.3f}): {hist[i]}")

    meta_path = out_dir / "meta.json"
    meta.update({
        "n_train": len(train_data),
        "n_val": len(val_data),
        "skip_parse": skip_parse,
        "skip_eval": skip_eval,
        "skip_no_moves": skip_no_moves,
        "skip_dup": skip_dup,
        "K": K,
        "CP_CLIP": CP_CLIP,
        "sampling": "streaming_fallback",
    })
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Updated {meta_path}")
    print("\n=== Build complete (streaming fallback) ===")


# ---------------------------------------------------------------------------
# Loaders (section 5.3)
# ---------------------------------------------------------------------------
def get_loaders(
    data_dir: str | Path, batch_size: int
) -> tuple:
    """Load .npy arrays and return (train_loader, val_loader).

    Each loader yields ``(tokens: LongTensor[B,64], aux: LongTensor[B],
    target: FloatTensor[B])``.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    data_dir = Path(data_dir)

    def _load_split(name: str) -> DataLoader:
        tokens = np.load(data_dir / f"{name}_tokens.npy")
        aux = np.load(data_dir / f"{name}_aux.npy")
        targets = np.load(data_dir / f"{name}_targets.npy")

        ds = TensorDataset(
            torch.from_numpy(tokens.astype(np.int64)),
            torch.from_numpy(aux.astype(np.int64)),
            torch.from_numpy(targets),
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=0,
        )

    train_loader = _load_split("train")
    val_loader = _load_split("val")
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Selftest (section 9)
# ---------------------------------------------------------------------------
def cmd_selftest(args: argparse.Namespace) -> None:
    """Run encoding and conversion correctness checks."""
    passed = 0
    failed = 0

    def check(name: str, condition: bool, msg: str = ""):
        nonlocal passed, failed
        if condition:
            print(f"  [PASS] {name}")
            passed += 1
        else:
            print(f"  [FAIL] {name}: {msg}")
            failed += 1

    print("=== data_pipeline selftest ===\n")

    # --- Test 1: Start position encoding ---
    print("Test 1: Start position encoding")
    board = chess.Board()
    tokens, aux = encode_board(board)

    # White to move: rank 0 = white back rank (a1..h1)
    # Expected: R=4, N=2, B=3, Q=5, K=6, B=3, N=2, R=4
    expected_back = [4, 2, 3, 5, 6, 3, 2, 4]
    check("Own back rank (0-7)",
          list(tokens[0:8]) == expected_back,
          f"got {list(tokens[0:8])}")

    check("Own pawns (8-15)",
          all(tokens[i] == 1 for i in range(8, 16)),
          f"got {list(tokens[8:16])}")

    check("Empty middle (16-47)",
          all(tokens[i] == 0 for i in range(16, 48)),
          f"non-zero found in middle")

    check("Opp pawns (48-55)",
          all(tokens[i] == 7 for i in range(48, 56)),
          f"got {list(tokens[48:56])}")

    expected_opp_back = [10, 8, 9, 11, 12, 9, 8, 10]
    check("Opp back rank (56-63)",
          list(tokens[56:64]) == expected_opp_back,
          f"got {list(tokens[56:64])}")

    # aux: all castling rights → castling = 0b1111 = 15, no en passant → ep_code = 0
    # aux = 15 * 9 + 0 = 135
    check("Aux = 135", aux == 135, f"got {aux}")
    print()

    # --- Test 2: Fast vs reference on 1000 random positions ---
    print("Test 2: Fast vs reference on 1000 random positions")
    rng = random.Random(42)
    mismatch = 0
    mirror_mismatch = 0
    for i in range(1000):
        b = chess.Board()
        n_moves = rng.randint(0, 80)
        for _ in range(n_moves):
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(rng.choice(moves))

        tok_fast, aux_fast = encode_board(b)
        tok_ref, aux_ref = encode_board_reference(b)

        if not np.array_equal(tok_fast, tok_ref) or aux_fast != aux_ref:
            mismatch += 1
            if mismatch <= 3:
                print(f"    Mismatch at position {i}: FEN={b.fen()}")

        # Mirror test: encode_board(b) == encode_board(b.mirror())
        b_mirror = b.mirror()
        tok_m, aux_m = encode_board(b_mirror)
        if not np.array_equal(tok_fast, tok_m) or aux_fast != aux_m:
            mirror_mismatch += 1
            if mirror_mismatch <= 3:
                print(f"    Mirror mismatch at position {i}: FEN={b.fen()}")

    check("encode_board == encode_board_reference (1000 positions)",
          mismatch == 0, f"{mismatch} mismatches")
    check("encode_board(b) == encode_board(b.mirror()) (1000 positions)",
          mirror_mismatch == 0, f"{mirror_mismatch} mismatches")
    print()

    # --- Test 3: POV conversion ---
    print("Test 3: POV conversion")
    # White-POV cp = -900 with Black to move → STM cp = +900 → target > 0.9
    cp_white = -900
    cp_stm = -cp_white  # Black to move, negate
    target = cp_to_value(cp_stm)
    check("POV conversion: White-POV cp=-900, Black to move → target > 0.9",
          target > 0.9, f"target = {target:.4f}")

    # Round-trip
    cp_back = value_to_cp(cp_to_value(500.0))
    check("cp_to_value / value_to_cp round-trip (500cp)",
          abs(cp_back - 500.0) < 1.0, f"got {cp_back:.2f}")
    print()

    # Summary
    print(f"=== Selftest: {passed} passed, {failed} failed ===")
    if failed > 0:
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="data_pipeline",
        description="Chess position dataset builder.",
    )
    sub = parser.add_subparsers(dest="command")

    # inspect
    p_inspect = sub.add_parser("inspect", help="Inspect the HF dataset")
    p_inspect.add_argument("--hf-dataset", required=True, help="HuggingFace dataset repo id")
    p_inspect.add_argument("--fen-col", default=None, help="FEN column name")
    p_inspect.add_argument("--eval-col", default=None, help="Eval column name")
    p_inspect.add_argument("--out", default="data", help="Output directory")

    # build
    p_build = sub.add_parser("build", help="Build train/val dataset")
    p_build.add_argument("--hf-dataset", required=True, help="HuggingFace dataset repo id")
    p_build.add_argument("--n-train", type=int, default=1_000_000, help="Number of training samples")
    p_build.add_argument("--n-val", type=int, default=20_000, help="Number of validation samples")
    p_build.add_argument("--out", default="data", help="Output directory")
    p_build.add_argument("--seed", type=int, default=42, help="Random seed")
    p_build.add_argument("--stride", type=int, default=10, help="Keep every Nth row from each block")

    # selftest
    sub.add_parser("selftest", help="Run encoding correctness checks")

    args = parser.parse_args()
    if args.command == "inspect":
        cmd_inspect(args)
    elif args.command == "build":
        cmd_build(args)
    elif args.command == "selftest":
        cmd_selftest(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
