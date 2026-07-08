#!/bin/bash
# Ayudante: recibe un FEN y un movetime(ms), devuelve la jugada del motor Rust.
# MIMOTOR_LMR=0 (sin LMR): configuracion con mejor ELO medido (~2178) en esta sesion.
FEN="$1"
MOVETIME="${2:-5000}"
BIN=~/mi-motor-rust/target/release/mi-motor-rust
printf "uci\nisready\nposition fen %s\ngo movetime %s\nquit\n" "$FEN" "$MOVETIME" | MIMOTOR_LMR=0 "$BIN" 2>&1 | grep "^bestmove" | awk '{print $2}'
