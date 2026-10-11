#!/usr/bin/env python3
"""File 3: Alpha-beta search with transposition table and eval batching.

CLI subcommands:
    selftest – run search correctness checks with MaterialEvaluator.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from enum import IntEnum

import chess
import chess.polyglot
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pickle
from pathlib import Path

from data_pipeline import K, cp_to_value, encode_board, value_to_cp

# ---------------------------------------------------------------------------
# Library-Based Opening Book (Openix & chess.polyglot)
# ---------------------------------------------------------------------------
class OpeningBook:
    """Opening book powered by the Openix chess openings library and chess.polyglot.

    Loads the complete Encyclopedia of Chess Openings (ECO A00-E99) from the
    Openix library, caches positions by Zobrist hash for instant lookups,
    and supports standard Polyglot (.bin) books via python-chess.
    """

    def __init__(self, bin_path: str | None = "data/performance.bin", cache_path: str = "data/openix_book.pkl"):
        self.bin_path = bin_path
        self.cache_path = cache_path
        self._book: dict[int, dict] = {}
        self._loaded = False
        self._polyglot_reader = None

    def load(self) -> None:
        if self._loaded:
            return
        self._loaded = True

        # 1. Polyglot grandmaster book probing via python-chess
        target_bin = self.bin_path or "data/performance.bin"
        if Path(target_bin).exists():
            try:
                self._polyglot_reader = chess.polyglot.open_reader(target_bin)
            except Exception as e:
                print(f"Warning: Failed to load Polyglot book {target_bin}: {e}")

        # 2. Try fast loading pre-compiled Openix book from cache
        if self.cache_path and Path(self.cache_path).exists():
            try:
                with open(self.cache_path, "rb") as f:
                    self._book = pickle.load(f)
                return
            except Exception:
                pass

        # 3. Load from Openix library (ECO A00-E99)
        try:
            import Openix
            lib = Openix.ChessOpeningsLibrary()
            lib.load_builtin_openings()
            for eco, openings in lib.openings.items():
                for op in openings:
                    b = chess.Board()
                    for m_str in op.moves_list:
                        z = chess.polyglot.zobrist_hash(b)
                        try:
                            m = b.parse_san(m_str)
                        except Exception:
                            try:
                                m = b.parse_uci(m_str)
                            except Exception:
                                break
                        if z not in self._book:
                            self._book[z] = {"moves": [], "name": op.name, "eco": op.eco_code}
                        uci = m.uci()
                        if uci not in self._book[z]["moves"]:
                            self._book[z]["moves"].append(uci)
                        b.push(m)

            # Save compiled book for instant startup next time
            try:
                Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_path, "wb") as f:
                    pickle.dump(self._book, f)
            except Exception:
                pass
        except ImportError:
            pass

    def get_entry(self, board: chess.Board) -> dict | None:
        """Lookup opening moves and opening information for a board position."""
        if not self._loaded:
            self.load()

        # Polyglot grandmaster weighted lookup
        if self._polyglot_reader is not None:
            try:
                entry = self._polyglot_reader.weighted_choice(board)
                if entry and entry.move in board.legal_moves:
                    return {"moves": [entry.move.uci()], "name": "Grandmaster Book", "eco": ""}
            except IndexError:
                pass  # Position not in Polyglot book
            except Exception:
                pass

        # Openix lookup via zobrist hash
        zhash = chess.polyglot.zobrist_hash(board)
        if zhash in self._book:
            legal = [
                uci for uci in self._book[zhash]["moves"]
                if chess.Move.from_uci(uci) in board.legal_moves
            ]
            if legal:
                return {
                    "moves": legal,
                    "name": self._book[zhash].get("name", "Opening"),
                    "eco": self._book[zhash].get("eco", ""),
                }
        return None


DEFAULT_OPENING_BOOK = OpeningBook()
OPENING_BOOK = DEFAULT_OPENING_BOOK

# ---------------------------------------------------------------------------
# Constants (section 7.1)
# ---------------------------------------------------------------------------
MATE = 10.0
DRAW = 0.0

DEFAULT_MAX_DEPTH = 4
DEFAULT_TIME_LIMIT = 15.0
TIME_CHECK_INTERVAL = 256

TT_MAX = 2_000_000
EVAL_CACHE_MAX = 1_000_000

# Piece values for MVV-LVA ordering
PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}


class TTFlag(IntEnum):
    EXACT = 0
    LOWER = 1  # alpha (fail-high)
    UPPER = 2  # beta (fail-low)


class SearchTimeout(Exception):
    """Raised when time limit is exceeded."""
    pass


# ---------------------------------------------------------------------------
# MaterialEvaluator (section 7.7)
# ---------------------------------------------------------------------------
class MaterialEvaluator:
    """Simple material-based evaluator for development and testing.

    Uses piece values P=100, N=320, B=330, R=500, Q=900.
    Computes from encoded tokens: own tokens 1-5 vs opponent tokens 7-11.
    Returns tanh(material_cp_stm / 400).
    """
    PIECE_CP = {1: 100, 2: 320, 3: 330, 4: 500, 5: 900}  # own pieces 1-5

    def evaluate_encoded(
        self, tokens: np.ndarray, aux: np.ndarray
    ) -> np.ndarray:
        """Evaluate from encoded tokens.

        Args:
            tokens: uint8 [n, 64]
            aux: int64 [n] (unused)

        Returns:
            float32 [n], values in (-1, 1).
        """
        n = len(tokens)
        results = np.empty(n, dtype=np.float32)
        for i in range(n):
            mat = 0
            for sq in range(64):
                t = int(tokens[i, sq])
                if 1 <= t <= 5:
                    mat += self.PIECE_CP[t]
                elif 7 <= t <= 11:
                    mat -= self.PIECE_CP[t - 6]
            results[i] = math.tanh(mat / K)
        return results

    def evaluate(self, boards: list) -> np.ndarray:
        """Evaluate a list of chess.Board objects."""
        n = len(boards)
        tokens = np.empty((n, 64), dtype=np.uint8)
        aux = np.empty(n, dtype=np.int64)
        for i, board in enumerate(boards):
            t, a = encode_board(board)
            tokens[i] = t
            aux[i] = a
        return self.evaluate_encoded(tokens, aux)


# ---------------------------------------------------------------------------
# Searcher (section 7)
# ---------------------------------------------------------------------------
class Searcher:
    """Negamax alpha-beta search with iterative deepening.

    Args:
        evaluator: An object with evaluate_encoded(tokens, aux) method.
        max_depth: Maximum search depth (default 4).
        time_limit: Seconds per move (default 15.0).
        use_qsearch: Enable quiescence search (default False).
    """

    def __init__(
        self,
        evaluator,
        max_depth: int = DEFAULT_MAX_DEPTH,
        time_limit: float = DEFAULT_TIME_LIMIT,
        use_qsearch: bool = True,
        use_book: bool = True,
        opening_book: OpeningBook | None = None,
    ):
        self.evaluator = evaluator
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.use_qsearch = use_qsearch
        self.use_book = use_book
        self.opening_book = opening_book if opening_book is not None else DEFAULT_OPENING_BOOK

        # Transposition table: hash → (depth, value, flag, best_move)
        self.tt: dict[int, tuple[int, float, int, chess.Move | None]] = {}
        # Eval cache: hash → float (STM value)
        self.eval_cache: dict[int, float] = {}
        # Killer moves: ply → [move, move]
        self.killers: dict[int, list[chess.Move]] = {}
        # History table: (from_sq, to_sq) → int
        self.history: dict[tuple[int, int], int] = {}

        # Stats
        self.nodes = 0
        self.evals = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self._start_time = 0.0

    def _check_time(self) -> None:
        if self.nodes % TIME_CHECK_INTERVAL == 0:
            if time.time() - self._start_time > self.time_limit:
                raise SearchTimeout()

    def _clear_tables_if_full(self) -> None:
        if len(self.tt) > TT_MAX:
            self.tt.clear()
        if len(self.eval_cache) > EVAL_CACHE_MAX:
            self.eval_cache.clear()

    # -----------------------------------------------------------------------
    # Leaf evaluation
    # -----------------------------------------------------------------------
    def _eval_single(self, board: chess.Board) -> float:
        """Evaluate a single position (depth 0 outside batching)."""
        zhash = chess.polyglot.zobrist_hash(board)
        if zhash in self.eval_cache:
            self.cache_hits += 1
            return self.eval_cache[zhash]

        self.cache_misses += 1
        tokens, aux = encode_board(board)
        val = self.evaluator.evaluate_encoded(
            tokens.reshape(1, 64), np.array([aux], dtype=np.int64)
        )[0]
        self.evals += 1
        self.eval_cache[zhash] = float(val)
        return float(val)

    def _is_terminal(self, board: chess.Board, ply: int) -> tuple[bool, float]:
        """Check if position is terminal. Returns (is_terminal, score)."""
        moves = list(board.legal_moves)
        if not moves:
            if board.is_check():
                return True, -(MATE - 0.01 * ply)
            return True, DRAW

        if ply > 0:
            if board.is_insufficient_material():
                return True, DRAW
            if board.is_fifty_moves():
                return True, DRAW
            if board.halfmove_clock >= 4 and board.is_repetition(2):
                return True, DRAW

        return False, 0.0

    # -----------------------------------------------------------------------
    # Move ordering (section 7.3)
    # -----------------------------------------------------------------------
    def _order_moves(
        self, board: chess.Board, moves: list[chess.Move],
        tt_move: chess.Move | None, ply: int,
    ) -> list[chess.Move]:
        """Order moves by priority: TT, MVV-LVA, killers, history."""
        scored = []
        killers = self.killers.get(ply, [])

        for move in moves:
            score = 0

            # TT / PV move
            if move == tt_move:
                score = 100_000
            # Captures and promotions
            elif board.is_capture(move):
                # MVV-LVA: 10 * victim - attacker
                victim_sq = move.to_square
                victim = board.piece_type_at(victim_sq)
                if victim is None:
                    # En passant
                    victim_val = 1
                else:
                    victim_val = PIECE_VALUES.get(victim, 0)

                attacker = board.piece_type_at(move.from_square)
                attacker_val = PIECE_VALUES.get(attacker, 0) if attacker else 0
                score = 50_000 + 10 * victim_val - attacker_val

                if move.promotion == chess.QUEEN:
                    score += 20_000
            elif move.promotion:
                score = 60_000 if move.promotion == chess.QUEEN else 40_000
            # Killers
            elif move in killers:
                score = 30_000
            # History
            else:
                score = self.history.get((move.from_square, move.to_square), 0)

            scored.append((score, move))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]

    def _record_killer(self, move: chess.Move, ply: int) -> None:
        if ply not in self.killers:
            self.killers[ply] = []
        killers = self.killers[ply]
        if move not in killers:
            killers.insert(0, move)
            if len(killers) > 2:
                killers.pop()

    def _bump_history(self, move: chess.Move, depth: int) -> None:
        key = (move.from_square, move.to_square)
        self.history[key] = self.history.get(key, 0) + depth * depth

    # -----------------------------------------------------------------------
    # Depth-1 batched evaluation (section 7.4)
    # -----------------------------------------------------------------------
    def _eval_depth1_batched(
        self, board: chess.Board, moves: list[chess.Move], ply: int,
    ) -> tuple[float, chess.Move]:
        """Evaluate all children at depth 1 in a single batch.

        Returns (best_score, best_move).
        """
        child_data = []  # (move, is_terminal, terminal_score, tokens, aux, zhash)

        for move in moves:
            board.push(move)
            try:
                self.nodes += 1
                self._check_time()

                # Check terminal
                terminal, term_score = self._is_terminal(board, ply + 1)
                if terminal:
                    child_data.append((move, True, term_score, None, None, None))
                    continue

                # If qsearch is enabled and child is not quiet (in check or has captures)
                if self.use_qsearch:
                    in_check = board.is_check()
                    has_captures = any(board.is_capture(m) or m.promotion for m in board.legal_moves)
                    if in_check or has_captures:
                        q_score = self._qsearch(board, -float("inf"), float("inf"), ply + 1, 0)
                        child_data.append((move, True, q_score, None, None, None))
                        continue

                # Encode and check cache
                tokens, aux = encode_board(board)
                zhash = chess.polyglot.zobrist_hash(board)

                if zhash in self.eval_cache:
                    self.cache_hits += 1
                    cached_val = self.eval_cache[zhash]
                    child_data.append((move, True, cached_val, None, None, None))
                    continue

                self.cache_misses += 1
                child_data.append((move, False, 0.0, tokens, aux, zhash))
            finally:
                board.pop()

        # Batch evaluate uncached positions
        uncached = [(i, d) for i, d in enumerate(child_data) if not d[1]]
        if uncached:
            batch_tokens = np.array([d[3] for _, d in uncached], dtype=np.uint8)
            batch_aux = np.array([d[4] for _, d in uncached], dtype=np.int64)
            vals = self.evaluator.evaluate_encoded(batch_tokens, batch_aux)
            self.evals += len(uncached)

            for j, (idx, d) in enumerate(uncached):
                move, _, _, _, _, zhash = d
                val = float(vals[j])
                self.eval_cache[zhash] = val
                child_data[idx] = (move, True, val, None, None, None)

        # Find best: child_value is from child's STM, parent_score = -child_value
        best_score = -float("inf")
        best_move = moves[0]
        for move, _, child_val, _, _, _ in child_data:
            parent_score = -child_val
            if parent_score > best_score:
                best_score = parent_score
                best_move = move

        return best_score, best_move

    # -----------------------------------------------------------------------
    # Negamax (section 7.2)
    # -----------------------------------------------------------------------
    def _negamax(
        self, board: chess.Board, depth: int, alpha: float, beta: float, ply: int,
    ) -> float:
        """Negamax alpha-beta search (fail-soft)."""
        self.nodes += 1
        self._check_time()
        self._clear_tables_if_full()

        # Step 1: Check for terminal
        moves = list(board.legal_moves)
        if not moves:
            if board.is_check():
                return -(MATE - 0.01 * ply)
            return DRAW

        # Step 2: Draw detection (ply > 0)
        if ply > 0:
            if board.is_insufficient_material():
                return DRAW
            if board.is_fifty_moves():
                return DRAW
            # Step 3: Repetition
            if board.halfmove_clock >= 4 and board.is_repetition(2):
                return DRAW

        # Step 4: TT probe
        zhash = chess.polyglot.zobrist_hash(board)
        tt_move = None
        if zhash in self.tt:
            tt_depth, tt_value, tt_flag, tt_best = self.tt[zhash]
            tt_move = tt_best

            if tt_depth >= depth and abs(tt_value) <= MATE / 2:
                if tt_flag == TTFlag.EXACT:
                    return tt_value
                elif tt_flag == TTFlag.LOWER and tt_value >= beta:
                    return tt_value
                elif tt_flag == TTFlag.UPPER and tt_value <= alpha:
                    return tt_value

        # Step 5: Leaf
        if depth == 0:
            if self.use_qsearch:
                return self._qsearch(board, alpha, beta, ply, 0)
            return self._eval_single(board)

        # Step 6: Order moves
        ordered_moves = self._order_moves(board, moves, tt_move, ply)

        # Step 7: Depth 1 → batched evaluation
        if depth == 1:
            best_score, best_move = self._eval_depth1_batched(board, ordered_moves, ply)
            # Store as EXACT
            if abs(best_score) <= MATE / 2:
                self.tt[zhash] = (depth, best_score, TTFlag.EXACT, best_move)
            else:
                self.tt[zhash] = (depth, 0.0, TTFlag.EXACT, best_move)
            return best_score

        # Step 8: Normal alpha-beta loop
        best_score = -float("inf")
        best_move = ordered_moves[0]
        original_alpha = alpha

        for i, move in enumerate(ordered_moves):
            board.push(move)
            try:
                score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            finally:
                board.pop()

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score

            if alpha >= beta:
                # Beta cutoff
                if not board.is_capture(move):
                    self._record_killer(move, ply)
                    self._bump_history(move, depth)
                break

        # Store in TT
        if abs(best_score) <= MATE / 2:
            if best_score <= original_alpha:
                flag = TTFlag.UPPER
            elif best_score >= beta:
                flag = TTFlag.LOWER
            else:
                flag = TTFlag.EXACT
            self.tt[zhash] = (depth, best_score, flag, best_move)
        else:
            # Don't store mate scores, but keep the move
            self.tt[zhash] = (depth, 0.0, TTFlag.EXACT, best_move)

        return best_score

    # -----------------------------------------------------------------------
    # Quiescence search (section 7.6, optional)
    # -----------------------------------------------------------------------
    def _qsearch(
        self, board: chess.Board, alpha: float, beta: float, ply: int,
        q_ply: int,
    ) -> float:
        """Quiescence search: captures and promotions only."""
        self.nodes += 1
        self._check_time()

        stand_pat = self._eval_single(board)

        if q_ply >= 4:
            return stand_pat

        in_check = board.is_check()

        if not in_check:
            if stand_pat >= beta:
                return stand_pat
            if stand_pat > alpha:
                alpha = stand_pat

        # Generate moves
        if in_check:
            moves = list(board.legal_moves)
        else:
            moves = [m for m in board.legal_moves
                     if board.is_capture(m) or m.promotion]

        if not moves:
            if in_check:
                return -(MATE - 0.01 * ply)
            return stand_pat

        # Order by MVV-LVA
        moves = self._order_moves(board, moves, None, ply)

        best = stand_pat if not in_check else -float("inf")
        for move in moves:
            board.push(move)
            try:
                score = -self._qsearch(board, -beta, -alpha, ply + 1, q_ply + 1)
            finally:
                board.pop()

            if score > best:
                best = score
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break

        return best

    # -----------------------------------------------------------------------
    # Iterative deepening (section 7.5)
    # -----------------------------------------------------------------------
    def best_move(
        self, board: chess.Board
    ) -> tuple[chess.Move, float, dict]:
        """Find the best move using iterative deepening.

        Returns:
            (move, value, info_dict)
        """
        moves = list(board.legal_moves)
        if not moves:
            raise ValueError("No legal moves available")

        # Check opening book
        if self.use_book:
            entry = self.opening_book.get_entry(board)
            if entry and entry.get("moves"):
                chosen_uci = random.choice(entry["moves"])
                chosen = chess.Move.from_uci(chosen_uci)
                if chosen in board.legal_moves:
                    return chosen, 0.0, {
                        "depth": "book", "nodes": 1, "evals": 0, "nps": 0,
                        "cache_hit_rate": 0.0, "pv": [chosen], "seconds": 0.0,
                        "cp": 0.0, "mate_in": None,
                        "opening_name": entry.get("name"),
                        "opening_eco": entry.get("eco"),
                    }

        # Single legal move → return immediately
        if len(moves) == 1:
            board.push(moves[0])
            is_term, term_val = self._is_terminal(board, 1)
            if is_term:
                val = -term_val
            else:
                val = -self._eval_single(board)
            board.pop()
            return moves[0], val, {
                "depth": 0, "nodes": 1, "evals": 0,
                "nps": 0, "cache_hit_rate": 0.0, "pv": [moves[0]],
                "seconds": 0.0, "cp": value_to_cp(val) if abs(val) <= MATE / 2 else None,
                "mate_in": None,
            }

        # Reset stats
        self.nodes = 0
        self.evals = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.killers.clear()
        self.history.clear()
        self._start_time = time.time()

        best_move = moves[0]
        best_value = -float("inf")
        best_depth = 0
        root_scores: dict[chess.Move, float] = {}

        initial_stack_len = len(board.move_stack)
        try:
            for d in range(1, self.max_depth + 1):
                try:
                    # Order root moves by previous iteration's scores
                    if d > 1 and root_scores:
                        moves.sort(key=lambda m: root_scores.get(m, -float("inf")),
                                   reverse=True)

                    iter_best_move = moves[0]
                    iter_best_value = -float("inf")
                    iter_scores = {}
                    alpha = -float("inf")
                    beta = float("inf")

                    for move in moves:
                        board.push(move)
                        try:
                            score = -self._negamax(board, d - 1, -beta, -alpha, 1)
                        finally:
                            board.pop()

                        iter_scores[move] = score
                        if score > iter_best_value:
                            iter_best_value = score
                            iter_best_move = move
                        if score > alpha:
                            alpha = score

                    # Completed this depth
                    best_move = iter_best_move
                    best_value = iter_best_value
                    best_depth = d
                    root_scores = iter_scores

                except SearchTimeout:
                    # Discard incomplete iteration, use last completed depth
                    break
        finally:
            while len(board.move_stack) > initial_stack_len:
                board.pop()

        elapsed = time.time() - self._start_time
        nps = self.nodes / max(0.001, elapsed)

        total_cache = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / max(1, total_cache)

        # Build PV from TT
        pv = self._extract_pv(board, best_depth)

        # Format info
        info: dict = {
            "depth": best_depth,
            "nodes": self.nodes,
            "evals": self.evals,
            "nps": int(nps),
            "cache_hit_rate": cache_hit_rate,
            "pv": pv,
            "seconds": elapsed,
        }

        if abs(best_value) > MATE / 2:
            mate_ply = round((MATE - abs(best_value)) / 0.01)
            info["mate_in"] = math.ceil(mate_ply / 2)
            info["cp"] = None
            if best_value < 0:
                info["mate_in"] = -info["mate_in"]
        else:
            info["cp"] = value_to_cp(best_value)
            info["mate_in"] = None

        return best_move, best_value, info

    def _extract_pv(self, board: chess.Board, max_len: int) -> list[chess.Move]:
        """Extract PV from transposition table."""
        pv = []
        seen = set()
        b = board.copy()

        for _ in range(max_len):
            zhash = chess.polyglot.zobrist_hash(b)
            if zhash in seen:
                break
            seen.add(zhash)

            entry = self.tt.get(zhash)
            if entry is None:
                break

            _, _, _, tt_move = entry
            if tt_move is None or tt_move not in b.legal_moves:
                break

            pv.append(tt_move)
            b.push(tt_move)

        return pv


# ---------------------------------------------------------------------------
# Plain minimax (for selftest comparison)
# ---------------------------------------------------------------------------
def _plain_minimax(board: chess.Board, evaluator, depth: int, ply: int) -> float:
    """Plain minimax without pruning, TT, or batching. For testing only."""
    moves = list(board.legal_moves)
    if not moves:
        if board.is_check():
            return -(MATE - 0.01 * ply)
        return DRAW

    if ply > 0:
        if board.is_insufficient_material():
            return DRAW
        if board.is_fifty_moves():
            return DRAW
        if board.halfmove_clock >= 4 and board.is_repetition(2):
            return DRAW

    if depth == 0:
        tokens, aux = encode_board(board)
        val = evaluator.evaluate_encoded(
            tokens.reshape(1, 64), np.array([aux], dtype=np.int64)
        )[0]
        return float(val)

    best = -float("inf")
    for move in moves:
        board.push(move)
        score = -_plain_minimax(board, evaluator, depth - 1, ply + 1)
        board.pop()
        if score > best:
            best = score
    return best


# ---------------------------------------------------------------------------
# Selftest (section 9)
# ---------------------------------------------------------------------------
def cmd_selftest(args: argparse.Namespace) -> None:
    """Run search correctness checks."""
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

    evaluator = MaterialEvaluator()
    print("=== search selftest (MaterialEvaluator) ===\n")

    # --- Test 1: Mate in 1 positions ---
    print("Test 1: Mate-in-1 positions")

    # Position from spec
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1")
    searcher = Searcher(evaluator, max_depth=2, time_limit=60.0)
    move, value, info = searcher.best_move(board)
    check("Ra8# found",
          board.san(move) == "Ra8#",
          f"got {board.san(move)}")

    # Scholar's mate threat: queen delivers mate
    board2 = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
    # Qxf7# is mate
    searcher2 = Searcher(evaluator, max_depth=2, time_limit=60.0)
    move2, value2, info2 = searcher2.best_move(board2)
    check("Qxf7# found (scholar's mate)",
          board2.san(move2) == "Qxf7#",
          f"got {board2.san(move2)}")

    # Mate-in-1 position 3: Back rank mate
    board3 = chess.Board("6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1")
    searcher3 = Searcher(evaluator, max_depth=2, time_limit=60.0)
    move3, val3, _ = searcher3.best_move(board3)
    check("Re8# found (back rank mate)",
          board3.san(move3) == "Re8#",
          f"got {board3.san(move3)}, val={val3}")
    print()

    # --- Mate in 2 ---
    print("Test 1b: Mate-in-2 positions")
    # Position 1: 1. Qxf7+ Kh8 2. Qh7#
    board_m2a = chess.Board("6k1/5p1p/8/6N1/8/5Q2/8/6K1 w - - 0 1")
    searcher_m2a = Searcher(evaluator, max_depth=4, time_limit=60.0)
    move_m2a, val_m2a, info_m2a = searcher_m2a.best_move(board_m2a)
    check("Mate-in-2 (Qxf7+ then mate) found",
          board_m2a.san(move_m2a) == "Qxf7#" or val_m2a > MATE / 2,
          f"move = {board_m2a.san(move_m2a)}, value = {val_m2a}")

    # Position 2: 1. Kg6 Kg8 2. Qb8#
    board_m2b = chess.Board("7k/8/5K2/8/8/8/8/1Q6 w - - 0 1")
    searcher_m2b = Searcher(evaluator, max_depth=4, time_limit=60.0)
    move_m2b, val_m2b, info_m2b = searcher_m2b.best_move(board_m2b)
    check("Mate-in-2 (King + Queen endgame) found",
          val_m2b > MATE / 2,
          f"move = {board_m2b.san(move_m2b)}, value = {val_m2b}")
    print()

    # --- Test 2: Alpha-beta vs plain minimax ---
    print("Test 2: Alpha-beta vs plain minimax (depth 3)")
    test_positions = [
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "r1bqkbnr/pppppppp/2n5/8/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 2 2",
        "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",
    ]

    ab_vs_mm_ok = True
    for fen in test_positions:
        board = chess.Board(fen)
        # Plain minimax
        mm_val = _plain_minimax(board, evaluator, 3, 0)
        # Alpha-beta
        searcher_ab = Searcher(evaluator, max_depth=3, time_limit=60.0, use_book=False, use_qsearch=False)
        _, ab_val, _ = searcher_ab.best_move(board)

        if abs(mm_val - ab_val) > 1e-6:
            ab_vs_mm_ok = False
            print(f"    MISMATCH: {fen}")
            print(f"      minimax={mm_val:.6f}, alpha-beta={ab_val:.6f}")

    check("Alpha-beta matches minimax at depth 3 (5 positions)", ab_vs_mm_ok)
    print()

    # --- Test 3: Draw positions ---
    print("Test 3: Draw positions")

    # K vs K
    board_kvk = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    searcher_kvk = Searcher(evaluator, max_depth=2, time_limit=60.0)
    move_kvk, val_kvk, _ = searcher_kvk.best_move(board_kvk)
    check("K vs K scores 0.0", abs(val_kvk) < 1e-6, f"value = {val_kvk}")

    # 50-move rule: halfmove_clock = 100
    board_50 = chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 100 1")
    searcher_50 = Searcher(evaluator, max_depth=2, time_limit=60.0)
    _, val_50, _ = searcher_50.best_move(board_50)
    check("halfmove_clock=100 (not mate) scores 0.0",
          abs(val_50) < 1e-6, f"value = {val_50}")

    # Repetition: create a position that has occurred before
    board_rep = chess.Board()
    # Play Nf3 Nf6 Ng1 Ng8 (back to start) then search
    rep_moves = ["g1f3", "g8f6", "f3g1", "f6g8"]
    for uci in rep_moves:
        board_rep.push(chess.Move.from_uci(uci))
    # Now the start position has occurred twice (initial + after 4 moves)
    searcher_rep = Searcher(evaluator, max_depth=2, time_limit=60.0)
    # At ply > 0, if a child leads to repetition, it should score 0
    # The root is at ply 0, children are at ply 1
    # Playing Nf3 again would create the position for the 2nd time at ply 1
    # which should score 0 via is_repetition(2)
    move_rep, val_rep, _ = searcher_rep.best_move(board_rep)
    # The move shouldn't be Nf3 (draw by repetition) if better exists
    check("Repetition detection works (returns legal move)",
          move_rep in board_rep.legal_moves)
    print()

    # --- Test 4: Queen trap ---
    print("Test 4: Don't take defended pawn with queen")
    # White queen on e2, black pawn on e5 defended by pawn on d6
    # Queen should not capture defended pawn Qxe5 at depth >= 2
    board_qt = chess.Board("rnbqkbnr/ppp2ppp/3p4/4p3/8/8/PPPPQPPP/RNB1KBNR w KQkq - 0 1")
    searcher_qt = Searcher(evaluator, max_depth=3, time_limit=60.0)
    move_qt, _, _ = searcher_qt.best_move(board_qt)
    # Qxe5 would lose the queen to dxe5
    qxe5 = chess.Move.from_uci("e2e5")
    check("Doesn't take defended pawn with queen (depth 3)",
          move_qt != qxe5,
          f"played {board_qt.san(move_qt)}")
    print()

    # --- Test 5: Time limit ---
    print("Test 5: Time limit respected")
    board_time = chess.Board()
    searcher_time = Searcher(evaluator, max_depth=20, time_limit=2.0)
    t0 = time.time()
    move_time, _, info_time = searcher_time.best_move(board_time)
    elapsed = time.time() - t0
    check("Returns within time_limit + 1.0s",
          elapsed < 3.0,
          f"took {elapsed:.1f}s")
    check("Returns a legal move",
          move_time in board_time.legal_moves)
    print()

    # Summary
    print(f"=== search selftest: {passed} passed, {failed} failed ===")
    if failed > 0:
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="search",
        description="Chess alpha-beta search.",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("selftest", help="Run search correctness checks")

    args = parser.parse_args()
    if args.command == "selftest":
        cmd_selftest(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
