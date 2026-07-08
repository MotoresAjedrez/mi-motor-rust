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

const MAX_KILLER_PLY: usize = 100; // margen sobre MAX_PLY para cubrir extensiones de jaque

pub struct Searcher {
    tt: Vec<Option<TTEntry>>,
    tt_mask: usize,
    pub nodes: u64,
    deadline: Option<Instant>,
    stop: bool,
    // killers son validos solo dentro de esta busqueda (por ply del arbol
    // actual); history SI persiste entre jugadas de la partida, igual que la TT.
    killers: Vec<[Option<Move>; 2]>,
    history: Box<[[i32; 64]; 64]>, // [from][to] -- arreglo plano, mas rapido que un HashMap aqui
    pub modo_lmr: bool,
    // Historial de repeticion: claves Zobrist de la PARTIDA REAL (persiste
    // entre llamadas a go, la maneja el loop UCI) + las de la linea actual
    // de busqueda (crece/decrece durante la recursion, como el "self.hist"
    // de Python). No se usa la TT para esto porque una entrada de TT no
    // sabe CUANTAS veces se visito esa posicion en esta partida especifica.
    game_history: Vec<u64>,
    path: Vec<u64>,
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
        Searcher {
            tt: vec![None; n_entries],
            tt_mask: n_entries - 1,
            nodes: 0,
            deadline: None,
            stop: false,
            killers: vec![[None, None]; MAX_KILLER_PLY],
            history: Box::new([[0i32; 64]; 64]),
            modo_lmr: std::env::var("MIMOTOR_LMR").as_deref() == Ok("1"),
            game_history: Vec::new(),
            path: Vec::new(),
        }
    }

    fn registrar_corte(&mut self, mv: Move, ply: u32, depth: i32) {
        if mv.is_capture() {
            return; // MVV-LVA/SEE ya ordenan las capturas primero, no necesitan refuerzo
        }
        let p = ply as usize;
        if p < MAX_KILLER_PLY {
            let k = &mut self.killers[p];
            if k[0] != Some(mv) {
                k[1] = k[0];
                k[0] = Some(mv);
            }
        }
        self.history[mv.from as usize][mv.to as usize] += depth * depth;
    }

    pub fn clear_tt(&mut self) {
        for e in self.tt.iter_mut() {
            *e = None;
        }
        self.history = Box::new([[0i32; 64]; 64]);
        self.game_history.clear();
    }

    /// Fija el historial de claves Zobrist de la PARTIDA REAL hasta la
    /// posicion actual (lo arma el loop UCI a partir de "position ...
    /// moves ..."). Se llama antes de cada busqueda para que la deteccion
    /// de repeticion vea jugadas ya ocurridas en la partida, no solo las
    /// que aparezcan dentro del arbol de esta busqueda.
    pub fn set_game_history(&mut self, hist: Vec<u64>) {
        self.game_history = hist;
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
        self.order_moves_ply(b, moves, tt_move, MAX_KILLER_PLY as u32);
    }

    /// Igual que order_moves pero ademas usa killers/history (por ply) para
    /// ordenar las jugadas silenciosas -- capturas/TT siguen mandando.
    fn order_moves_ply(&self, b: &Board, moves: &mut Vec<Move>, tt_move: Option<Move>, ply: u32) {
        let p = ply as usize;
        let killers = if p < MAX_KILLER_PLY { Some(self.killers[p]) } else { None };
        moves.sort_by_key(|mv| {
            if Some(*mv) == tt_move {
                return -1_000_000;
            }
            if mv.is_capture() {
                -(10_000 + crate::see::see(b, mv))
            } else if mv.promotion.is_some() {
                -5000
            } else if killers.is_some_and(|k| k[0] == Some(*mv)) {
                -3000
            } else if killers.is_some_and(|k| k[1] == Some(*mv)) {
                -2900
            } else {
                -self.history[mv.from as usize][mv.to as usize]
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
            // Filtro SEE: descarta capturas claramente perdedoras (no las
            // prueba ni gasta tiempo en ellas). Margen -50, no se aplica a
            // promociones (la poda delta de abajo ya las valora aparte).
            if mv.promotion.is_none() && crate::see::see(b, &mv) < -50 {
                continue;
            }
            let victim = if mv.flag == MoveFlag::EnPassant {
                100
            } else {
                b.piece_at(mv.to).map(|(_, pt)| valor_pieza(pt)).unwrap_or(0)
            };
            let mut ganancia = victim;
            if mv.promotion.is_some() {
                ganancia += 800;
            }
            if stand_pat + ganancia + 250 <= alpha {
                continue; // poda delta
            }
            let next = b.make_move(&mv);
            if next.in_check(b.turn) {
                continue; // ilegal: propio rey quedaria en jaque
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

        // Repeticion: si esta posicion ya aparecio entre los ancestros
        // (partida real + linea de busqueda actual) dentro de la ventana de
        // jugadas reversibles (halfmove_clock), tratarla como tablas -- asi
        // el motor las evita activamente cuando esta mejor y las busca
        // activamente cuando esta peor, en vez de solo "no perder por
        // descuido". No hace falta esperar la 3ra ocurrencia real: ver la
        // 2da dentro del arbol ya significa que la repeticion esta
        // disponible como opcion, que es lo que le interesa a la busqueda.
        let hc = b.halfmove_clock as usize;
        if hc > 0 {
            let start = self.path.len().saturating_sub(hc);
            if self.path[start..].contains(&b.zobrist) {
                return Ok(0);
            }
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
        self.order_moves_ply(b, &mut moves, tt_move, ply);
        self.path.push(b.zobrist);

        // Mas conservador que en Python (que reducia desde la jugada #3 a
        // partir de profundidad 3): con SEE+killers el motor en Rust ya
        // llega mucho mas hondo que Python en el mismo segundo de reloj
        // (nps 100-500x mayor), asi que "ganar una ply mas" vale bastante
        // menos y el riesgo de descartar una jugada buena en la
        // verificacion reducida pesa mas. Medido: la version "Python-like"
        // (desde jugada 3, prof>=3, reduccion de hasta 2 ply) le costo
        // ~320 ELO en el torneo de referencia (18 partidas) pese a haber
        // dado bien en un mini-torneo de 4 partidas -- reducir desde mas
        // tarde en el orden, a mas profundidad, y nunca mas de 1 ply.
        const LMR_MOVES_SIN_REDUCIR: usize = 5;
        const LMR_PROF_MIN: i32 = 5;

        let mut best_score = -INFINITO;
        let mut best_move = None;
        for (idx, mv) in moves.iter().enumerate() {
            // LMR: candidatas a reducir son jugadas silenciosas, tarde en el
            // orden (ya viene de mejor a peor), sin jaque propio ni jaque
            // que dan -- justo donde el orden ya filtra la mayoria de
            // jugadas malas sin gastar profundidad completa.
            let es_reducible = self.modo_lmr
                && !en_jaque
                && idx >= LMR_MOVES_SIN_REDUCIR
                && depth >= LMR_PROF_MIN
                && !mv.is_capture()
                && mv.promotion.is_none();

            let next = b.make_move(mv);
            let sc = if es_reducible && !next.in_check(next.turn) {
                let r = 1i32.min(depth - 2);
                let reducido = -self.negamax(&next, depth - 1 - r, -beta, -alpha, ply + 1)?;
                if reducido > alpha {
                    // la reduccion sugiere que podria ser buena: confirmar a profundidad completa
                    -self.negamax(&next, depth - 1, -beta, -alpha, ply + 1)?
                } else {
                    reducido
                }
            } else {
                -self.negamax(&next, depth - 1, -beta, -alpha, ply + 1)?
            };

            if sc > best_score {
                best_score = sc;
                best_move = Some(*mv);
            }
            if sc > alpha {
                alpha = sc;
            }
            if alpha >= beta {
                self.registrar_corte(*mv, ply, depth);
                break;
            }
        }
        self.path.pop();

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
        self.killers = vec![[None, None]; MAX_KILLER_PLY];
        self.path = self.game_history.clone();
        let mut mejor_mv = None;
        let mut mejor_sc = -INFINITO;
        for d in 1..=depth {
            let moves = generate_legal(b);
            if moves.is_empty() {
                break;
            }
            let mut ordered = moves.clone();
            self.order_moves_ply(b, &mut ordered, mejor_mv, 0);
            let mut alpha = -INFINITO;
            let mut actual_mv = ordered[0];
            let mut actual_sc = -INFINITO;
            self.path.push(b.zobrist);
            for mv in &ordered {
                let next = b.make_move(mv);
                let sc = match self.negamax(&next, d - 1, -INFINITO, -alpha, 1) {
                    Ok(v) => -v,
                    Err(_) => {
                        self.path.pop();
                        return (mejor_mv.or(Some(actual_mv)), mejor_sc, self.nodes);
                    }
                };
                if sc > actual_sc {
                    actual_sc = sc;
                    actual_mv = *mv;
                }
                if sc > alpha {
                    alpha = sc;
                }
            }
            self.path.pop();
            mejor_mv = Some(actual_mv);
            mejor_sc = actual_sc;
        }
        (mejor_mv, mejor_sc, self.nodes)
    }

    /// Busqueda con presupuesto de tiempo (para UCI "go movetime").
    pub fn search_time(&mut self, b: &Board, movetime_ms: u64, max_depth: i32, mut on_info: impl FnMut(i32, i32, u64, u64)) -> (Option<Move>, i32) {
        self.nodes = 0;
        self.stop = false;
        self.killers = vec![[None, None]; MAX_KILLER_PLY];
        self.path = self.game_history.clone();
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
            self.order_moves_ply(b, &mut ordered, mejor_mv, 0);
            let mut alpha = -INFINITO;
            let mut actual_mv = ordered[0];
            let mut actual_sc = -INFINITO;
            let mut timed_out = false;
            self.path.push(b.zobrist);
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
            self.path.pop();
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
