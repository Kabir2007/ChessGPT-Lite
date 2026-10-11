#!/usr/bin/env python3
"""File 4: CLI game loop for playing against the chess bot.

Usage:
    python play.py [--color W|B] [--checkpoint checkpoints/best.pt]
                   [--material] [--depth 4] [--time 15] [--fen "<fen>"]
                   [--threads N]

CLI subcommands:
    (default) – play a game
    selftest  – run input parsing and game loop checks.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import chess
import chess.pgn

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from data_pipeline import value_to_cp
from model import Evaluator
from search import MATE, MaterialEvaluator, Searcher


# ---------------------------------------------------------------------------
# Move parsing (section 8.3)
# ---------------------------------------------------------------------------
def parse_user_move(board: chess.Board, text: str) -> chess.Move:
    """Parse user input into a chess.Move.

    Accepts SAN (Nxc6, exd5, O-O) and UCI (e2e4, e7e8q).
    Handles castling normalization and promotion prompting.

    Raises ValueError with a human-readable message on failure.
    """
    # Step 1: Strip whitespace, do NOT lowercase
    text = text.strip()
    if not text:
        raise ValueError("Empty input. Type a move or 'help' for commands.")

    # Step 2: Castling normalization: 0-0 → O-O, 0-0-0 → O-O-O
    if re.match(r'^[0O]-[0O](-[0O])?[+#]?$', text, re.IGNORECASE):
        text = text.replace('0', 'O')

    # Step 3: Strip trailing +, #, !, ?
    cleaned = re.sub(r'[+#!?]+$', '', text)

    # Step 4: Try SAN first
    try:
        return board.parse_san(cleaned)
    except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError):
        pass

    # Try UCI
    try:
        return board.parse_uci(cleaned)
    except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError):
        pass

    # Step 5: Promotion detection – pawn move to last rank without promotion piece
    # Check if appending =Q (or another piece) would work
    last_rank = 7 if board.turn == chess.WHITE else 0
    # Try adding promotion suffix
    for promo_char in ['q', 'r', 'b', 'n']:
        # UCI style: e7e8q
        try:
            uci_with_promo = cleaned + promo_char
            move = board.parse_uci(uci_with_promo)
            # This is a valid promotion move – ask user what piece
            promo = _prompt_promotion()
            final_text = cleaned + promo.lower()
            return board.parse_uci(final_text)
        except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError):
            pass

        # SAN style: e8=Q, exd8=Q
        try:
            san_with_promo = cleaned + '=' + promo_char.upper()
            move = board.parse_san(san_with_promo)
            promo = _prompt_promotion()
            final_text = cleaned + '=' + promo
            return board.parse_san(final_text)
        except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError):
            pass

    raise ValueError(
        f"'{text}' is not a legal move. Type 'moves' to see legal moves."
    )


def _prompt_promotion() -> str:
    """Prompt the user for a promotion piece."""
    while True:
        try:
            choice = input("Promote to (Q/R/B/N)? ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            print()
            raise ValueError("Cancelled")
        if choice in ('Q', 'R', 'B', 'N'):
            return choice
        print("Invalid. Choose Q, R, B, or N.")


# ---------------------------------------------------------------------------
# Board display
# ---------------------------------------------------------------------------
def print_board(board: chess.Board, human_color: chess.Color, force: bool = False) -> None:
    """Print the board only if force is True (e.g. when user explicitly types 'board')."""
    if force:
        orientation = human_color
        print()
        print(board.unicode(orientation=orientation))
        print()


def format_eval_white_pov(value: float, stm: chess.Color, info: dict) -> str:
    """Format evaluation from White's perspective."""
    mate_in = info.get("mate_in")
    if mate_in is not None:
        # mate_in is from STM's perspective
        if stm == chess.BLACK:
            mate_in = -mate_in
        sign = "+" if mate_in > 0 else ""
        return f"M{sign}{mate_in}"

    # Convert to White's perspective
    cp = info.get("cp")
    if cp is None:
        cp = value_to_cp(value)

    if stm == chess.BLACK:
        cp = -cp

    pawns = cp / 100.0
    sign = "+" if pawns >= 0 else ""
    return f"{sign}{pawns:.2f}"


# ---------------------------------------------------------------------------
# Game loop (section 8.2)
# ---------------------------------------------------------------------------
def play_game(
    board: chess.Board,
    searcher: Searcher,
    evaluator,
    bot_color: chess.Color,
    human_color: chess.Color,
) -> None:
    """Main game loop."""
    print("\n♟ ChessGPT – type 'help' for commands\n")

    while True:
        # Check for game over
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            print(f"Game over: {outcome.result()} ({outcome.termination.name})")
            _print_pgn(board)
            return

        if board.turn == bot_color:
            # Bot's turn
            move, value, info = searcher.best_move(board)
            san = board.san(move)
            board.push(move)

            eval_str = format_eval_white_pov(value, bot_color, info)
            print(f"Bot plays: {san}")
            print(f"  depth={info['depth']} eval={eval_str} "
                  f"nodes={info['nodes']} time={info['seconds']:.1f}s")

            # Check outcome after bot move
            outcome = board.outcome(claim_draw=True)
            if outcome is not None:
                print(f"Game over: {outcome.result()} ({outcome.termination.name})")
                _print_pgn(board)
                return
        else:
            # Human's turn
            while True:
                try:
                    text = input("Your move: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye!")
                    return

                if not text:
                    continue

                # Commands
                cmd = text.lower()
                if cmd == "help":
                    _print_help()
                    continue
                elif cmd == "moves":
                    _print_legal_moves(board)
                    continue
                elif cmd == "board":
                    print_board(board, human_color, force=True)
                    continue
                elif cmd == "fen":
                    print(f"FEN: {board.fen()}")
                    continue
                elif cmd == "eval":
                    _print_eval(board, evaluator)
                    continue
                elif cmd == "undo":
                    if len(board.move_stack) < 2:
                        print("Cannot undo: fewer than 2 moves on the stack.")
                        continue
                    board.pop()
                    board.pop()
                    print("Undone 2 plies.")
                    continue
                elif cmd == "resign":
                    winner = "White" if bot_color == chess.WHITE else "Black"
                    print(f"You resign. {winner} (bot) wins.")
                    _print_pgn(board)
                    return
                elif cmd == "quit":
                    print("Goodbye!")
                    return

                # Parse move
                try:
                    move = parse_user_move(board, text)
                    board.push(move)
                    break
                except ValueError as e:
                    print(f"Error: {e}")
                    continue


def _print_help() -> None:
    print("Commands:")
    print("  help   – show this help")
    print("  moves  – list legal moves")
    print("  board  – reprint the board")
    print("  fen    – print the current FEN")
    print("  eval   – print model evaluation")
    print("  undo   – undo last move pair")
    print("  resign – resign the game")
    print("  quit   – exit")


def _print_legal_moves(board: chess.Board) -> None:
    moves = [board.san(m) for m in board.legal_moves]
    print(f"Legal moves ({len(moves)}): {', '.join(moves)}")


def _print_eval(board: chess.Board, evaluator) -> None:
    vals = evaluator.evaluate([board])
    v = float(vals[0])
    cp = value_to_cp(v)
    print(f"  STM value: {v:+.4f} ({cp:+.0f} cp)")


def _print_pgn(board: chess.Board) -> None:
    game = chess.pgn.Game.from_board(board)
    print(f"\nPGN:\n{game}\n")


# ---------------------------------------------------------------------------
# Selftest (section 9)
# ---------------------------------------------------------------------------
def cmd_selftest(args: argparse.Namespace) -> None:
    """Run input parsing and game loop checks."""
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

    print("=== play selftest ===\n")

    # --- Test 1: parse_user_move accepts valid moves ---
    print("Test 1: parse_user_move accepts valid inputs")

    board = chess.Board()
    # Standard opening moves
    check("e4 (SAN)",
          parse_user_move(board, "e4") == chess.Move.from_uci("e2e4"))
    check("e2e4 (UCI)",
          parse_user_move(board, "e2e4") == chess.Move.from_uci("e2e4"))
    check("Nf3 (SAN)",
          parse_user_move(board, "Nf3") == chess.Move.from_uci("g1f3"))

    # Castling
    # Set up a position where O-O is legal
    board_castle = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
    check("O-O (SAN)",
          parse_user_move(board_castle, "O-O") == chess.Move.from_uci("e1g1"))
    check("0-0 (zero notation)",
          parse_user_move(board_castle, "0-0") == chess.Move.from_uci("e1g1"))
    check("O-O-O (queenside)",
          parse_user_move(board_castle, "O-O-O") == chess.Move.from_uci("e1c1"))

    # Capture
    board_cap = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2")
    check("exd5 (capture)",
          parse_user_move(board_cap, "exd5") == chess.Move.from_uci("e4d5"))

    # Knight capture
    board_nxc = chess.Board("r1bqkbnr/pppppppp/2n5/8/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 2")
    # Nxc6 is not actually legal here (wrong square), let's use a valid one
    # Actually from this position, there's no Nxc6. Let's set up one.
    board_nxc = chess.Board("r1bqkb1r/pppppppp/2n2n2/4N3/4P3/8/PPPP1PPP/RNBQKB1R w KQkq - 0 1")
    if chess.Move.from_uci("e5c6") in board_nxc.legal_moves:
        check("Nxc6 (knight capture)",
              parse_user_move(board_nxc, "Nxc6") == chess.Move.from_uci("e5c6"))
    else:
        # Find any valid Nxc6 position
        board_nxc2 = chess.Board("r1bqkbnr/pppp1ppp/2n5/4N3/4P3/8/PPPP1PPP/RNBQKB1R w KQkq - 0 1")
        if chess.Move.from_uci("e5c6") in board_nxc2.legal_moves:
            check("Nxc6 (knight capture)",
                  parse_user_move(board_nxc2, "Nxc6") == chess.Move.from_uci("e5c6"))
        else:
            print("  ⊘ Nxc6 test skipped (couldn't find valid position)")

    # Promotion with explicit piece
    board_promo = chess.Board("8/4P3/8/8/8/8/4k3/4K3 w - - 0 1")
    if chess.Move.from_uci("e7e8q") in board_promo.legal_moves:
        check("e8=Q (promotion SAN)",
              parse_user_move(board_promo, "e8=Q") == chess.Move.from_uci("e7e8q"))
    print()

    # --- Test 2: parse_user_move rejects invalid inputs ---
    print("Test 2: parse_user_move rejects invalid inputs")

    def expect_error(name, board, text):
        try:
            parse_user_move(board, text)
            check(name, False, "did not raise ValueError")
        except ValueError:
            check(name, True)

    expect_error("Illegal move (e5 from start)", chess.Board(), "e5")
    expect_error("Garbage text", chess.Board(), "xyz123")
    expect_error("Ambiguous (if applicable)", chess.Board(), "Nge2")  # not legal from start
    print()

    # --- Test 3: Scripted game through loop ---
    print("Test 3: Scripted game scenarios")

    # En passant
    board_ep = chess.Board("rnbqkbnr/pppp1ppp/8/4pP2/8/8/PPPPP1PP/RNBQKBNR w KQkq e6 0 3")
    if chess.Move.from_uci("f5e6") in board_ep.legal_moves:
        check("En passant (fxe6)",
              parse_user_move(board_ep, "fxe6") == chess.Move.from_uci("f5e6"))

    # Threefold repetition ending
    board_rep = chess.Board()
    moves = ["g1f3", "g8f6", "f3g1", "f6g8",
             "g1f3", "g8f6", "f3g1", "f6g8"]
    for uci in moves:
        board_rep.push(chess.Move.from_uci(uci))
    outcome = board_rep.outcome(claim_draw=True)
    check("Threefold repetition detected",
          outcome is not None and outcome.termination == chess.Termination.THREEFOLD_REPETITION,
          f"outcome={outcome}")

    # 50-move rule
    board_50 = chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 99 1")
    board_50.push(chess.Move.from_uci("a1a2"))  # halfmove_clock = 100
    outcome_50 = board_50.outcome(claim_draw=True)
    check("50-move rule detected",
          outcome_50 is not None and outcome_50.termination == chess.Termination.FIFTY_MOVES,
          f"outcome={outcome_50}")

    # Checkmate
    board_mate = chess.Board("rnbqkbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 1")
    # Not quite right, fool's mate needs: 1.f3 e5 2.g4 Qh4#
    board_mate = chess.Board()
    for uci in ["f2f3", "e7e5", "g2g4", "d8h4"]:
        board_mate.push(chess.Move.from_uci(uci))
    outcome_mate = board_mate.outcome()
    check("Checkmate detected (fool's mate)",
          outcome_mate is not None and outcome_mate.termination == chess.Termination.CHECKMATE,
          f"outcome={outcome_mate}")

    # Stalemate
    board_stale = chess.Board("k7/8/1K6/8/8/8/8/8 b - - 0 1")
    # This might not be stalemate. Let's use a known one.
    board_stale = chess.Board("k7/8/2K5/8/8/8/8/1Q6 b - - 0 1")
    # Check if stalemate
    if not list(board_stale.legal_moves):
        outcome_stale = board_stale.outcome()
        check("Stalemate detected",
              outcome_stale is not None and outcome_stale.termination == chess.Termination.STALEMATE,
              f"outcome={outcome_stale}")
    else:
        # Use a verified stalemate position
        board_stale = chess.Board("k7/8/1K6/8/8/8/8/8 b - - 0 1")
        if not list(board_stale.legal_moves) and not board_stale.is_check():
            outcome_stale = board_stale.outcome()
            check("Stalemate detected",
                  outcome_stale is not None,
                  f"outcome={outcome_stale}")
        else:
            print("  ⊘ Stalemate test using bare king position")
            # Verified stalemate: Black king on a8, White king on a6, White queen on b6
            board_stale = chess.Board("k7/8/KQ6/8/8/8/8/8 b - - 0 1")
            outcome_stale = board_stale.outcome()
            check("Stalemate detected",
                  outcome_stale is not None and outcome_stale.termination == chess.Termination.STALEMATE,
                  f"outcome={outcome_stale}")
    print()

    # Summary
    print(f"=== play selftest: {passed} passed, {failed} failed ===")
    if failed > 0:
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="play",
        description="Play chess against the bot.",
    )
    sub = parser.add_subparsers(dest="command")

    # selftest
    sub.add_parser("selftest", help="Run input parsing checks")

    # Default: play
    parser.add_argument("--color", type=str, default=None,
                        help="Your color (W or B)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt",
                        help="Model checkpoint path")
    parser.add_argument("--material", action="store_true",
                        help="Use material evaluator instead of model")
    parser.add_argument("--depth", type=int, default=4,
                        help="Search depth")
    parser.add_argument("--time", type=float, default=15.0,
                        help="Time limit per move (seconds)")
    parser.add_argument("--fen", type=str, default=None,
                        help="Starting FEN position")
    parser.add_argument("--threads", type=int, default=None,
                        help="Number of threads")
    parser.add_argument("--no-qsearch", dest="qsearch", action="store_false", default=True,
                        help="Disable quiescence search (enabled by default for tactical stability)")

    args = parser.parse_args()

    if args.command == "selftest":
        cmd_selftest(args)
        return

    # Set threads
    if args.threads:
        import torch
        torch.set_num_threads(args.threads)

    # Determine human color (--color W means User plays White)
    user_color_str = args.color
    if user_color_str is None:
        while True:
            try:
                user_color_str = input("Play as White or Black (W/B)? ").strip().upper()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                return
            if user_color_str in ('W', 'B'):
                break
            print("Please enter W or B.")

    human_color = chess.WHITE if user_color_str.upper() == 'W' else chess.BLACK
    bot_color = not human_color

    # Set up evaluator
    if args.material:
        evaluator = MaterialEvaluator()
        print("Using MaterialEvaluator")
    else:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            print(f"Error: checkpoint not found: {ckpt_path}")
            print("Use --material flag to play with material evaluator.")
            sys.exit(1)
        evaluator = Evaluator.from_checkpoint(ckpt_path)
        print(f"Using model from {ckpt_path}")

    # Set up board
    if args.fen:
        board = chess.Board(args.fen)
    else:
        board = chess.Board()

    # Set up searcher
    searcher = Searcher(
        evaluator,
        max_depth=args.depth,
        time_limit=args.time,
        use_qsearch=args.qsearch,
    )

    # Play
    try:
        play_game(board, searcher, evaluator, bot_color, human_color)
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
