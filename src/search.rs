// Negamax + poda alfa-beta + iterative deepening + quiescence + TT.
// Primera version jugable de la Fase 3: SEE, null-move, killers/history y
// LMR quedan para una siguiente pasada si el tiempo alcanza (documentado
// en el reporte final de la sesion).

use crate::board::Board;
use crate::eval::evaluate;
use crate::movegen::{generate_legal, generate_pseudo_legal};
use crate::types::{Move, MoveFlag};
use std::time::Instant;

pub const INFINITO: i32 = 30_000;
pub const MATE: i32 = 29_000;
const MAX_PLY: u32 = 64;

fn solo_peones_y_rey(b: &Board, color: crate::types::Color) -> bool {
    let idx = color as usize;
    (b.pieces[idx][crate::types::PieceType::Knight as usize]
        | b.pieces[idx][crate::types::PieceType::Bishop as usize]
        | b.pieces[idx][crate::types::PieceType::Rook as usize]
        | b.pieces[idx][crate::types::PieceType::Queen as usize])
        == 0
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TTFlag {
    Exact,
    Alpha,
    Beta,
}

#[derive(Clone, Copy)]
struct TTEntry {
    key: u64,
    depth: i32,
    score: i32,
    flag: TTFlag,
    best: Option<Move>,
}

pub struct TimeUp;

pub struct Searcher {
    tt: Vec<Option<TTEntry>>,
    tt_mask: usize,
    pub nodes: u64,
    deadline: Option<Instant>,
    stop: bool,
}

fn valor_pieza(pt: crate::types::PieceType) -> i32 {
    match pt {
        crate::types::PieceType::Pawn => 100,
        crate::types::PieceType::Knight => 320,
        crate::types::PieceType::Bishop => 330,
        crate::types::PieceType::Rook => 500,
        crate::types::PieceType::Queen => 900,
        crate::types::PieceType::King => 20000,
    }
}

impl Searcher {
    pub fn new(tt_mb: usize) -> Searcher {
        let entry_size = std::mem::size_of::<Option<TTEntry>>().max(1);
        let mut n_entries = (tt_mb * 1024 * 1024 / entry_size).max(1024);
        n_entries = n_entries.next_power_of_two() >> 1; // asegurar potencia de 2 sin pasarse
        Searcher { tt: vec![None; n_entries], tt_mask: n_entries - 1, nodes: 0, deadline: None, stop: false }
    }

    pub fn clear_tt(&mut self) {
        for e in self.tt.iter_mut() {
            *e = None;
        }
    }

    fn tt_index(&self, key: u64) -> usize {
        (key as usize) & self.tt_mask
    }

    fn tt_probe(&self, key: u64) -> Option<TTEntry> {
        match self.tt[self.tt_index(key)] {
            Some(e) if e.key == key => Some(e),
            _ => None,
        }
    }

    fn tt_store(&mut self, key: u64, depth: i32, score: i32, flag: TTFlag, best: Option<Move>) {
        let idx = self.tt_index(key);
        self.tt[idx] = Some(TTEntry { key, depth, score, flag, best });
    }

    fn check_time(&mut self) -> Result<(), TimeUp> {
        self.nodes += 1;
        if let Some(dl) = self.deadline {
            if self.nodes & 1023 == 0 && Instant::now() >= dl {
                self.stop = true;
            }
        }
        if self.stop {
            Err(TimeUp)
        } else {
            Ok(())
        }
    }

    fn order_moves(&self, b: &Board, moves: &mut Vec<Move>, tt_move: Option<Move>) {
        moves.sort_by_key(|mv| {
            if Some(*mv) == tt_move {
                return -1_000_000;
            }
            if mv.is_capture() {
                let victim = if mv.flag == MoveFlag::EnPassant {
                    100
                } else {
                    b.piece_at(mv.to).map(|(_, pt)| valor_pieza(pt)).unwrap_or(0)
                };
                let attacker = b.piece_at(mv.from).map(|(_, pt)| valor_pieza(pt)).unwrap_or(0);
                -(10_000 + victim * 16 - attacker)
            } else if mv.promotion.is_some() {
                -5000
            } else {
                0
            }
        });
    }

    fn quiescence(&mut self, b: &Board, mut alpha: i32, beta: i32, ply: u32) -> Result<i32, TimeUp> {
        self.check_time()?;
        let stand_pat = evaluate(b);
        if ply >= MAX_PLY {
            return Ok(stand_pat);
        }
        if stand_pat >= beta {
            return Ok(beta);
        }
        if stand_pat > alpha {
            alpha = stand_pat;
        }

        let mut moves: Vec<Move> = generate_pseudo_legal(b).into_iter().filter(|m| m.is_capture()).collect();
        self.order_moves(b, &mut moves, None);

        let mut best = stand_pat;
        for mv in moves {
            let next = b.make_move(&mv);
            if next.in_check(b.turn) {
                continue; // ilegal: propio rey quedaria en jaque
            }
            let victim = if mv.flag == MoveFlag::EnPassant {
                100
            } else {
                b.piece_at(mv.to).map(|(_, pt)| valor_pieza(pt)).unwrap_or(0)
            };
            if stand_pat + victim + 200 <= alpha {
                continue; // poda delta
            }
            let sc = -self.quiescence(&next, -beta, -alpha, ply + 1)?;
            if sc > best {
                best = sc;
            }
            if sc > alpha {
                alpha = sc;
            }
            if alpha >= beta {
                break;
            }
        }
        Ok(best)
    }

    fn negamax(&mut self, b: &Board, mut depth: i32, mut alpha: i32, beta: i32, ply: u32) -> Result<i32, TimeUp> {
        self.check_time()?;

        if b.halfmove_clock >= 100 {
            return Ok(0);
        }

        let en_jaque = b.in_check(b.turn);
        if en_jaque && ply < 40 {
            depth += 1; // extension de jaque
        }

        if depth <= 0 || ply >= MAX_PLY {
            return self.quiescence(b, alpha, beta, ply);
        }

        let alpha_orig = alpha;
        let key = b.zobrist;
        let mut tt_move = None;
        if let Some(entry) = self.tt_probe(key) {
            tt_move = entry.best;
            if entry.depth >= depth {
                match entry.flag {
                    TTFlag::Exact => return Ok(entry.score),
                    TTFlag::Beta if entry.score >= beta => return Ok(entry.score),
                    TTFlag::Alpha if entry.score <= alpha => return Ok(entry.score),
                    _ => {}
                }
            }
        }

        // Null-move pruning: si "pasar el turno" y aun asi el rival no supera
        // beta, la posicion ya es tan buena que se poda sin generar jugadas
        // reales. Desactivado en jaque, en finales de solo peones (riesgo de
        // zugzwang) y cerca de puntajes de mate (poco fiable ahi).
        const NULL_MOVE_R: i32 = 2;
        const NULL_MOVE_PROF_MIN: i32 = 3;
        if !en_jaque
            && depth >= NULL_MOVE_PROF_MIN
            && beta < MATE - 1000
            && alpha > -(MATE - 1000)
            && !solo_peones_y_rey(b, b.turn)
        {
            let next = b.make_null_move();
            let sc_null = -self.negamax(&next, depth - 1 - NULL_MOVE_R, -beta, -beta + 1, ply + 1)?;
            if sc_null >= beta {
                return Ok(beta);
            }
        }

        let mut moves = generate_legal(b);
        if moves.is_empty() {
            return Ok(if en_jaque { -MATE + ply as i32 } else { 0 });
        }
        self.order_moves(b, &mut moves, tt_move);

        let mut best_score = -INFINITO;
        let mut best_move = None;
        for mv in &moves {
            let next = b.make_move(mv);
            let sc = -self.negamax(&next, depth - 1, -beta, -alpha, ply + 1)?;
            if sc > best_score {
                best_score = sc;
                best_move = Some(*mv);
            }
            if sc > alpha {
                alpha = sc;
            }
            if alpha >= beta {
                break;
            }
        }

        let flag = if best_score <= alpha_orig {
            TTFlag::Alpha
        } else if best_score >= beta {
            TTFlag::Beta
        } else {
            TTFlag::Exact
        };
        self.tt_store(key, depth, best_score, flag, best_move);

        Ok(best_score)
    }

    /// Busqueda con profundidad fija (para benchmarks/tests, sin limite de tiempo).
    pub fn search_fixed_depth(&mut self, b: &Board, depth: i32) -> (Option<Move>, i32, u64) {
        self.nodes = 0;
        self.deadline = None;
        self.stop = false;
        let mut mejor_mv = None;
        let mut mejor_sc = -INFINITO;
        for d in 1..=depth {
            let moves = generate_legal(b);
            if moves.is_empty() {
                break;
            }
            let mut ordered = moves.clone();
            self.order_moves(b, &mut ordered, mejor_mv);
            let mut alpha = -INFINITO;
            let mut actual_mv = ordered[0];
            let mut actual_sc = -INFINITO;
            for mv in &ordered {
                let next = b.make_move(mv);
                let sc = match self.negamax(&next, d - 1, -INFINITO, -alpha, 1) {
                    Ok(v) => -v,
                    Err(_) => return (mejor_mv.or(Some(actual_mv)), mejor_sc, self.nodes),
                };
                if sc > actual_sc {
                    actual_sc = sc;
                    actual_mv = *mv;
                }
                if sc > alpha {
                    alpha = sc;
                }
            }
            mejor_mv = Some(actual_mv);
            mejor_sc = actual_sc;
        }
        (mejor_mv, mejor_sc, self.nodes)
    }

    /// Busqueda con presupuesto de tiempo (para UCI "go movetime").
    pub fn search_time(&mut self, b: &Board, movetime_ms: u64, max_depth: i32, mut on_info: impl FnMut(i32, i32, u64, u64)) -> (Option<Move>, i32) {
        self.nodes = 0;
        self.stop = false;
        let inicio = Instant::now();
        let budget = movetime_ms.saturating_sub(30).max(10);
        self.deadline = Some(inicio + std::time::Duration::from_millis(budget));

        let mut mejor_mv: Option<Move> = None;
        let mut mejor_sc = 0;

        for d in 1..=max_depth {
            let moves = generate_legal(b);
            if moves.is_empty() {
                break;
            }
            let mut ordered = moves.clone();
            self.order_moves(b, &mut ordered, mejor_mv);
            let mut alpha = -INFINITO;
            let mut actual_mv = ordered[0];
            let mut actual_sc = -INFINITO;
            let mut timed_out = false;
            for mv in &ordered {
                let next = b.make_move(mv);
                match self.negamax(&next, d - 1, -INFINITO, -alpha, 1) {
                    Ok(v) => {
                        let sc = -v;
                        if sc > actual_sc {
                            actual_sc = sc;
                            actual_mv = *mv;
                        }
                        if sc > alpha {
                            alpha = sc;
                        }
                    }
                    Err(_) => {
                        timed_out = true;
                        break;
                    }
                }
            }
            if timed_out {
                break;
            }
            mejor_mv = Some(actual_mv);
            mejor_sc = actual_sc;
            on_info(d, mejor_sc, self.nodes, inicio.elapsed().as_millis() as u64);

            if mejor_sc.abs() >= MATE - 1000 {
                break;
            }
            if inicio.elapsed().as_millis() as u64 > movetime_ms * 45 / 100 {
                break;
            }
        }
        (mejor_mv, mejor_sc)
    }
}
