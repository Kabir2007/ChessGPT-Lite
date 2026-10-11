#!/usr/bin/env python3
"""File 2: Encoder-only transformer model, training loop, and Evaluator.

CLI subcommands:
    train    – train the model on pre-built .npy data.
    bench    – measure training and inference throughput.
    selftest – run architecture and training correctness checks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Import from our data_pipeline
from data_pipeline import (
    K,
    CP_CLIP,
    cp_to_value,
    encode_board,
    get_loaders,
    value_to_cp,
)

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "d_model": 128,
    "n_layers": 4,
    "n_heads": 4,
    "d_ff": 512,
    "piece_vocab": 13,    # 0=empty, 1-6=STM, 7-12=opponent
    "aux_vocab": 144,     # castling(0-15) * 9 + ep_code(0-8)
    "seq_len": 65,        # 1 global + 64 squares
    "n_squares": 64,
    "dropout": 0.0,
    "attn_bias": False,
}


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    """Pre-LayerNorm transformer encoder block."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.0, attn_bias: bool = False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

        self.attn_bias = attn_bias
        if attn_bias:
            # Learned per-head additive attention bias [n_heads, 65, 65]
            self.bias = nn.Parameter(torch.zeros(n_heads, 65, 65))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LayerNorm
        normed = self.ln1(x)
        if self.attn_bias:
            # Use F.scaled_dot_product_attention via attn_mask
            attn_out, _ = self.attn(
                normed, normed, normed, attn_mask=self.bias.repeat(
                    x.size(0), 1, 1)  # [B*n_heads, 65, 65]
            )
        else:
            attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.dropout(attn_out)

        # FFN
        normed = self.ln2(x)
        ff_out = self.ff(normed)
        x = x + self.dropout(ff_out)
        return x


# ---------------------------------------------------------------------------
# ChessTransformer
# ---------------------------------------------------------------------------
class ChessTransformer(nn.Module):
    """Encoder-only transformer for chess position evaluation.

    Input:  tokens [B, 64] (int), aux [B] (int)
    Output: v [B] in (-1, 1)
    """

    def __init__(self, config: dict | None = None):
        super().__init__()
        cfg = {**DEFAULT_CONFIG, **(config or {})}
        self.config = cfg

        d = cfg["d_model"]

        # Embeddings
        self.piece_emb = nn.Embedding(cfg["piece_vocab"], d)
        self.square_pos_emb = nn.Embedding(cfg["n_squares"], d)
        self.cls_emb = nn.Parameter(torch.randn(d) * 0.02)
        self.aux_emb = nn.Embedding(cfg["aux_vocab"], d)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d, cfg["n_heads"], cfg["d_ff"],
                             cfg["dropout"], cfg["attn_bias"])
            for _ in range(cfg["n_layers"])
        ])

        # Output head: take index 0 → LN → Linear → GELU → Linear → tanh
        self.head_ln = nn.LayerNorm(d)
        self.head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: LongTensor [B, 64] – piece tokens
            aux: LongTensor [B] – aux castling/ep value

        Returns:
            v: Tensor [B] in (-1, 1)
        """
        B = tokens.size(0)
        d = self.config["d_model"]

        # Square token embeddings: piece_emb + pos_emb
        sq_indices = torch.arange(64, device=tokens.device).unsqueeze(0).expand(B, -1)
        square_embs = self.piece_emb(tokens) + self.square_pos_emb(sq_indices)  # [B, 64, d]

        # Global token: cls_emb + aux_emb
        global_token = self.cls_emb.unsqueeze(0).expand(B, -1) + self.aux_emb(aux)  # [B, d]
        global_token = global_token.unsqueeze(1)  # [B, 1, d]

        # Sequence: [global, sq0, sq1, ..., sq63] → [B, 65, d]
        x = torch.cat([global_token, square_embs], dim=1)

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        # Output head: take index 0
        cls_out = x[:, 0, :]  # [B, d]
        cls_out = self.head_ln(cls_out)
        v = self.head(cls_out).squeeze(-1)  # [B]
        v = torch.tanh(v)
        return v


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int,
                     min_lr: float = 1e-5):
    """Warmup + cosine decay scheduler."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr / 1e-3, cosine)  # min_lr / base_lr
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def validate(model: ChessTransformer, val_loader, device: torch.device, max_batches: int | None = None) -> dict:
    """Compute evaluation metrics over a data loader."""
    model.eval()
    total_mse = 0.0
    total_mae_cp = 0.0
    sign_correct = 0
    sign_total = 0
    n_samples = 0

    with torch.inference_mode():
        for b_idx, (tokens, aux, targets) in enumerate(val_loader):
            if max_batches is not None and b_idx >= max_batches:
                break
            tokens, aux, targets = tokens.to(device), aux.to(device), targets.to(device)
            preds = model(tokens, aux)

            mse = F.mse_loss(preds, targets, reduction="sum").item()
            total_mse += mse
            n_samples += len(targets)

            # Vectorized metric computation in NumPy (fast)
            preds_np = preds.cpu().numpy()
            targets_np = targets.cpu().numpy()

            preds_cp = K * np.arctanh(np.clip(preds_np, -0.9999, 0.9999))
            targets_cp = K * np.arctanh(np.clip(targets_np, -0.9999, 0.9999))
            total_mae_cp += float(np.sum(np.abs(preds_cp - targets_cp)))

            # Sign agreement for |cp| > 100
            mask = np.abs(targets_cp) > 100
            sign_total += int(np.sum(mask))
            sign_correct += int(np.sum((preds_cp[mask] > 0) == (targets_cp[mask] > 0)))

    metrics = {
        "val_mse": total_mse / max(1, n_samples),
        "mse": total_mse / max(1, n_samples),
        "mae_cp": total_mae_cp / max(1, n_samples),
        "sign_agreement": sign_correct / max(1, sign_total) if sign_total > 0 else 0.0,
        "sign_correct": sign_correct,
        "sign_total": sign_total,
        "n_samples": n_samples,
    }
    model.train()
    return metrics


def cmd_train(args: argparse.Namespace) -> None:
    """Train the model."""
    # Thread control
    n_threads = args.threads or max(1, os.cpu_count() // 2)
    torch.set_num_threads(n_threads)
    print(f"Using {n_threads} threads")

    device = torch.device("cpu")

    # Load data
    train_loader, val_loader = get_loaders(args.data, args.batch_size)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    config = dict(DEFAULT_CONFIG)
    if args.attn_bias:
        config["attn_bias"] = True
    model = ChessTransformer(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=0.01,
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = get_lr_scheduler(optimizer, warmup_steps=1000, total_steps=total_steps)

    # Resume
    start_step = 0
    best_val_mse = float("inf")
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt.get("step", 0)
        best_val_mse = ckpt.get("val_metrics", {}).get("val_mse", float("inf"))
        print(f"  Resumed at step {start_step}, best val MSE = {best_val_mse:.6f}")

    # Checkpoint directory
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(path: str | Path, step: int, val_metrics: dict, train_metrics: dict | None = None):
        torch.save({
            "state_dict": model.state_dict(),
            "config": config,
            "K": K,
            "CP_CLIP": CP_CLIP,
            "step": step,
            "val_metrics": val_metrics,
            "train_metrics": train_metrics,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }, path)

    # Training loop
    model.train()
    step = start_step
    log_loss = 0.0
    log_count = 0
    t_start = time.time()
    val_metrics = {}

    try:
        for epoch in range(args.epochs):
            print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
            for tokens, aux, targets in train_loader:
                tokens, aux, targets = tokens.to(device), aux.to(device), targets.to(device)

                preds = model(tokens, aux)
                loss = F.mse_loss(preds, targets)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                step += 1
                log_loss += loss.item()
                log_count += 1

                # Log every 100 steps
                if step % 100 == 0:
                    elapsed = time.time() - t_start
                    sps = (log_count * args.batch_size) / max(0.001, elapsed)
                    avg_loss = log_loss / log_count
                    lr = scheduler.get_last_lr()[0]
                    print(f"  step {step:6d} | loss {avg_loss:.6f} | lr {lr:.2e} | {sps:.0f} samples/sec")
                    log_loss = 0.0
                    log_count = 0
                    t_start = time.time()

                # Validate and checkpoint every 2000 steps
                if step % 2000 == 0:
                    val_metrics = validate(model, val_loader, device)
                    print(f"  [VAL] step {step} | mse {val_metrics['val_mse']:.6f} "
                          f"| mae_cp {val_metrics['mae_cp']:.1f} "
                          f"| sign_agree {val_metrics['sign_agreement']:.3f} "
                          f"({val_metrics['sign_total']} samples)")

                    save_checkpoint(ckpt_dir / "last.pt", step, val_metrics)
                    if val_metrics["val_mse"] < best_val_mse:
                        best_val_mse = val_metrics["val_mse"]
                        save_checkpoint(ckpt_dir / "best.pt", step, val_metrics)
                        print(f"  → New best model saved (val_mse = {best_val_mse:.6f})")

            # End of epoch evaluation for BOTH train and validation
            print(f"\nComputing end-of-epoch evaluation for Epoch {epoch + 1}/{args.epochs}...")
            train_eval_batches = getattr(args, "eval_train_batches", None)
            train_metrics = validate(model, train_loader, device, max_batches=train_eval_batches)
            val_metrics = validate(model, val_loader, device)

            print("\n" + "=" * 80)
            print(f"  EPOCH {epoch + 1}/{args.epochs} EVALUATION SUMMARY (TRAIN & VALIDATION)")
            print("=" * 80)
            print(f"  {'METRIC':<20} | {'TRAIN':<25} | {'VALIDATION':<25}")
            print("-" * 80)
            print(f"  {'Samples':<20} | {train_metrics['n_samples']:<25,d} | {val_metrics['n_samples']:<25,d}")
            print(f"  {'Loss (MSE)':<20} | {train_metrics['mse']:<25.6f} | {val_metrics['mse']:<25.6f}")
            print(f"  {'MAE (Centipawns)':<20} | {train_metrics['mae_cp']:<22.1f} cp | {val_metrics['mae_cp']:<22.1f} cp")
            print(f"  {'Sign Agreement':<20} | {train_metrics['sign_agreement'] * 100:<5.1f}% ({train_metrics['sign_correct']:,}/{train_metrics['sign_total']:,}) | {val_metrics['sign_agreement'] * 100:<5.1f}% ({val_metrics['sign_correct']:,}/{val_metrics['sign_total']:,})")
            print("=" * 80 + "\n")

            save_checkpoint(ckpt_dir / "last.pt", step, val_metrics, train_metrics)
            if val_metrics["val_mse"] < best_val_mse:
                best_val_mse = val_metrics["val_mse"]
                save_checkpoint(ckpt_dir / "best.pt", step, val_metrics, train_metrics)
                print(f"  → New best model saved (val_mse = {best_val_mse:.6f})\n")

    except KeyboardInterrupt:
        print("\n\n[NOTICE] Training interrupted by user (Ctrl+C). Saving current checkpoint...")
        if val_metrics:
            save_checkpoint(ckpt_dir / "last.pt", step, val_metrics)
        print(f"Checkpoint saved to {ckpt_dir / 'last.pt'}. Best val MSE recorded: {best_val_mse:.6f}")
        return

    print(f"\nTraining complete. {step} steps total. Best val MSE: {best_val_mse:.6f}")


# ---------------------------------------------------------------------------
# Bench
# ---------------------------------------------------------------------------
def cmd_bench(args: argparse.Namespace) -> None:
    """Measure training and inference throughput."""
    n_threads = args.threads or max(1, os.cpu_count() // 2)
    torch.set_num_threads(n_threads)
    print(f"Using {n_threads} threads")

    device = torch.device("cpu")
    model = ChessTransformer(DEFAULT_CONFIG).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Training throughput
    print("\n--- Training throughput ---")
    batch_size = args.batch_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()

    tokens = torch.randint(0, 13, (batch_size, 64))
    aux = torch.randint(0, 144, (batch_size,))
    targets = torch.randn(batch_size)

    # Warmup
    for _ in range(5):
        preds = model(tokens, aux)
        loss = F.mse_loss(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Timed
    n_iters = 50
    t0 = time.time()
    for _ in range(n_iters):
        preds = model(tokens, aux)
        loss = F.mse_loss(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    elapsed = time.time() - t0
    train_sps = (n_iters * batch_size) / elapsed
    print(f"  {train_sps:.0f} samples/sec (batch_size={batch_size})")

    # Inference throughput
    print("\n--- Inference throughput ---")
    model.eval()
    for bs in [1, 64, 512]:
        tokens = torch.randint(0, 13, (bs, 64))
        aux = torch.randint(0, 144, (bs,))

        # Warmup
        with torch.inference_mode():
            for _ in range(3):
                model(tokens, aux)

        n_iters = max(10, 200 // max(1, bs))
        t0 = time.time()
        with torch.inference_mode():
            for _ in range(n_iters):
                model(tokens, aux)
        elapsed = time.time() - t0
        eps = (n_iters * bs) / elapsed
        print(f"  batch_size={bs:4d}: {eps:.0f} evals/sec")

    # Estimated wall time
    if args.data and Path(args.data).exists():
        try:
            meta_path = Path(args.data) / "meta.json"
            with open(meta_path) as f:
                meta = json.load(f)
            n_train = meta.get("n_train", 300_000)
            epochs = args.epochs
            steps = (n_train // batch_size) * epochs
            est_seconds = steps * (batch_size / train_sps)
            est_minutes = est_seconds / 60
            print(f"\n--- Estimated wall time ---")
            print(f"  {n_train:,} samples × {epochs} epochs = {steps:,} steps")
            print(f"  ~{est_minutes:.1f} minutes ({est_seconds / 3600:.1f} hours)")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Evaluator (section 6.4)
# ---------------------------------------------------------------------------
class Evaluator:
    """Wraps a trained ChessTransformer for inference."""

    def __init__(self, model: ChessTransformer, device: torch.device, k: float = K):
        self.model = model
        self.device = device
        self.k = k
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> Evaluator:
        """Load an Evaluator from a checkpoint file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        config = ckpt.get("config", DEFAULT_CONFIG)
        k = ckpt.get("K", K)

        model = ChessTransformer(config)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return cls(model, torch.device("cpu"), k)

    def evaluate_encoded(
        self, tokens: np.ndarray, aux: np.ndarray
    ) -> np.ndarray:
        """Evaluate pre-encoded positions.

        Args:
            tokens: uint8 array of shape [n, 64]
            aux: int64 array of shape [n]

        Returns:
            float32 array of shape [n], values in (-1, 1) from STM perspective.
        """
        n = len(tokens)
        results = np.empty(n, dtype=np.float32)
        chunk_size = 512

        with torch.inference_mode():
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                t = torch.from_numpy(tokens[start:end].astype(np.int64))
                a = torch.from_numpy(aux[start:end].astype(np.int64))
                v = self.model(t.to(self.device), a.to(self.device))
                results[start:end] = v.cpu().numpy()

        return results

    def evaluate(self, boards: list) -> np.ndarray:
        """Evaluate a list of chess.Board objects.

        Returns:
            float32 array of shape [n], values in (-1, 1) from STM perspective.
        """
        n = len(boards)
        tokens = np.empty((n, 64), dtype=np.uint8)
        aux = np.empty(n, dtype=np.int64)

        for i, board in enumerate(boards):
            t, a = encode_board(board)
            tokens[i] = t
            aux[i] = a

        return self.evaluate_encoded(tokens, aux)


# ---------------------------------------------------------------------------
# Selftest (section 9)
# ---------------------------------------------------------------------------
def cmd_selftest(args: argparse.Namespace) -> None:
    """Run model correctness checks."""
    import chess

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

    print("=== model selftest ===\n")

    n_threads = max(1, os.cpu_count() // 2)
    torch.set_num_threads(n_threads)

    device = torch.device("cpu")
    model = ChessTransformer(DEFAULT_CONFIG).to(device)

    # --- Test 1: Forward shape ---
    print("Test 1: Forward shape and range")
    tokens = torch.randint(0, 13, (8, 64))
    aux = torch.randint(0, 144, (8,))
    with torch.inference_mode():
        out = model(tokens, aux)
    check("Output shape [B]", out.shape == (8,), f"got {out.shape}")
    check("Output in (-1, 1)", bool((out > -1).all() and (out < 1).all()),
          f"range [{out.min():.4f}, {out.max():.4f}]")
    print()

    # --- Test 2: Overfit gate ---
    print("Test 2: Overfit gate (2000 samples, up to 1000 steps)")
    if args.data and Path(args.data).exists():
        train_loader, _ = get_loaders(args.data, 256)
        # Get first 2000 samples
        all_tokens, all_aux, all_targets = [], [], []
        for t, a, tgt in train_loader:
            all_tokens.append(t)
            all_aux.append(a)
            all_targets.append(tgt)
            if sum(len(x) for x in all_tokens) >= 2000:
                break
        tokens_2k = torch.cat(all_tokens)[:2000]
        aux_2k = torch.cat(all_aux)[:2000]
        targets_2k = torch.cat(all_targets)[:2000]

        overfit_model = ChessTransformer(DEFAULT_CONFIG).to(device)
        optimizer = torch.optim.AdamW(overfit_model.parameters(), lr=1e-3)
        overfit_model.train()

        initial_loss = None
        final_loss = None
        bs = 256
        steps_taken = 0
        for step in range(1000):
            steps_taken = step + 1
            idx = torch.randint(0, 2000, (bs,))
            preds = overfit_model(tokens_2k[idx], aux_2k[idx])
            loss = F.mse_loss(preds, targets_2k[idx])

            if step == 0:
                initial_loss = loss.item()
            final_loss = loss.item()

            if initial_loss is not None and initial_loss / max(1e-12, final_loss) >= 10.0:
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        ratio = initial_loss / max(1e-12, final_loss)
        check("Loss drops by ≥ 10× in 1000 steps",
              ratio >= 10.0,
              f"initial={initial_loss:.6f}, final={final_loss:.6f}, ratio={ratio:.1f}× (at step {steps_taken})")
    else:
        print("  SKIP: --data not provided or doesn't exist")
    print()

    # --- Test 3: Mirror symmetry ---
    print("Test 3: Evaluator mirror symmetry")
    model.eval()
    evaluator = Evaluator(model, device)

    board = chess.Board()
    mirror = board.mirror()
    vals = evaluator.evaluate([board])
    vals_m = evaluator.evaluate([mirror])
    check("evaluate(b) == evaluate(b.mirror()) for start position",
          abs(vals[0] - vals_m[0]) < 1e-5,
          f"diff = {abs(vals[0] - vals_m[0]):.6f}")

    # A few more positions
    test_fens = [
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "r1bqkb1r/pppppppp/2n2n2/8/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    ]
    mirror_ok = True
    for fen in test_fens:
        b = chess.Board(fen)
        m = b.mirror()
        v1 = evaluator.evaluate([b])
        v2 = evaluator.evaluate([m])
        if abs(v1[0] - v2[0]) > 1e-5:
            mirror_ok = False
    check("Mirror symmetry on multiple positions", mirror_ok)
    print()

    # --- Test 4: Checkpoint round-trip ---
    print("Test 4: Checkpoint save/load round-trip")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        torch.save({
            "state_dict": model.state_dict(),
            "config": DEFAULT_CONFIG,
            "K": K,
            "CP_CLIP": CP_CLIP,
            "step": 0,
            "val_metrics": {},
            "optimizer": {},
            "scheduler": {},
        }, tmp_path)

        loaded = Evaluator.from_checkpoint(tmp_path)
        tokens_test = torch.randint(0, 13, (4, 64))
        aux_test = torch.randint(0, 144, (4,))
        with torch.inference_mode():
            orig_out = model(tokens_test, aux_test).numpy()
            load_out = loaded.model(tokens_test, aux_test).numpy()

        check("Checkpoint round-trip reproduces same outputs",
              np.allclose(orig_out, load_out, atol=1e-6),
              f"max diff = {np.max(np.abs(orig_out - load_out)):.8f}")
    finally:
        os.unlink(tmp_path)
    print()

    # Summary
    print(f"=== model selftest: {passed} passed, {failed} failed ===")
    if failed > 0:
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="model",
        description="Chess position evaluation transformer.",
    )
    sub = parser.add_subparsers(dest="command")

    # train
    p_train = sub.add_parser("train", help="Train the model")
    p_train.add_argument("--data", default="data", help="Data directory")
    p_train.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    p_train.add_argument("--batch-size", type=int, default=256, help="Batch size")
    p_train.add_argument("--threads", type=int, default=None, help="Number of threads")
    p_train.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    p_train.add_argument("--checkpoint-dir", default="checkpoints", help="Checkpoint directory")
    p_train.add_argument("--attn-bias", action="store_true", help="Enable learned attention bias")
    p_train.add_argument("--eval-train-batches", type=int, default=None, help="Number of training batches to evaluate at epoch end (default: None for full training set)")

    # bench
    p_bench = sub.add_parser("bench", help="Benchmark throughput")
    p_bench.add_argument("--data", default="data", help="Data directory")
    p_bench.add_argument("--batch-size", type=int, default=256, help="Batch size")
    p_bench.add_argument("--threads", type=int, default=None, help="Number of threads")
    p_bench.add_argument("--epochs", type=int, default=3, help="Epochs for time estimate")

    # selftest
    p_selftest = sub.add_parser("selftest", help="Run model checks")
    p_selftest.add_argument("--data", default="data", help="Data directory")

    args = parser.parse_args()
    if args.command == "train":
        cmd_train(args)
    elif args.command == "bench":
        cmd_bench(args)
    elif args.command == "selftest":
        cmd_selftest(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
