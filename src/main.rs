mod bitboard;
mod board;
mod eval;
mod movegen;
mod perft;
mod search;
mod see;
mod types;
mod zobrist;

use board::Board;
use search::Searcher;
use std::env;
use std::io::{self, BufRead, Write};
use std::time::Instant;
use types::{Move, MoveFlag, PieceType};

const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const KIWIPETE: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
const POSITION3: &str = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";
const POSITION5: &str = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 0 1";

fn run_perft_suite() {
    let cases: Vec<(&str, &str, Vec<u64>)> = vec![
        ("posicion inicial", STARTPOS, vec![1, 20, 400, 8902, 197281, 4865609, 119060324]),
        ("kiwipete", KIWIPETE, vec![1, 48, 2039, 97862, 4085603, 193690690]),
        ("posicion 3", POSITION3, vec![1, 14, 191, 2812, 43238, 674624, 11030083]),
        ("posicion 5", POSITION5, vec![1, 44, 1486, 62379, 2103487, 89941194]),
    ];

    let mut todo_ok = true;
    for (nombre, fen, esperados) in &cases {
        let b = Board::from_fen(fen).expect("FEN invalido en suite de perft");
        println!("\n=== {} ===", nombre);
        for (depth, &esperado) in esperados.iter().enumerate() {
            let t0 = Instant::now();
            let n = perft::perft(&b, depth as u32);
            let dt = t0.elapsed();
            let ok = n == esperado;
            todo_ok &= ok;
            let nps = if dt.as_secs_f64() > 0.0 { n as f64 / dt.as_secs_f64() } else { 0.0 };
            println!(
                "  depth {}: {} (esperado {}) {}  [{:.2}s, {:.0} nps]",
                depth, n, esperado,
                if ok { "OK" } else { "*** FALLO ***" },
                dt.as_secs_f64(), nps
            );
            if !ok {
                break;
            }
        }
    }
    println!("\n{}", if todo_ok { "TODOS LOS PERFT OK" } else { "HAY FALLOS DE PERFT -- NO AVANZAR A FASE 3" });
}

fn run_divide(fen: &str, depth: u32) {
    let b = Board::from_fen(fen).expect("FEN invalido");
    let mut total = 0u64;
    for (uci, n) in perft::perft_divide(&b, depth) {
        println!("{}: {}", uci, n);
        total += n;
    }
    println!("total: {}", total);
}

fn run_bench(depth: i32) {
    let posiciones = [
        ("inicial", STARTPOS),
        ("medio juego", "r1bqk2r/ppp2ppp/2n2n2/2bpp3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 6"),
    ];
    for (nombre, fen) in posiciones {
        let b = Board::from_fen(fen).unwrap();
        let mut s = Searcher::new(64);
        let t0 = Instant::now();
        let (mv, sc, nodes) = s.search_fixed_depth(&b, depth);
        let dt = t0.elapsed();
        let nps = nodes as f64 / dt.as_secs_f64().max(0.0001);
        println!(
            "{}: profundidad {} -> {} (score {}) | {} nodos en {:.2}s = {:.0} nps",
            nombre, depth, mv.map(|m| m.to_uci()).unwrap_or_default(), sc, nodes, dt.as_secs_f64(), nps
        );
    }
}

fn run_mate_tests() {
    let casos = [
        ("Mate en 1: Ta8# (pasillo)", "6k1/8/6K1/8/8/8/8/R7 w - - 0 1", "a1a8"),
        ("Mate en 1: Dh4# (mate del loco)", "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 1", "d8h4"),
        ("Mate en 1: Dxf7# (mate pastor)", "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", "h5f7"),
    ];
    let mut todo_ok = true;
    for (nombre, fen, esperada) in casos.iter() {
        let b = Board::from_fen(fen).unwrap();
        let mut s = Searcher::new(16);
        let (mv, sc, nodes) = s.search_fixed_depth(&b, 4);
        let uci = mv.map(|m| m.to_uci()).unwrap_or_default();
        let ok = uci == *esperada;
        todo_ok &= ok;
        println!(
            "{} {} -> {} (esperada {}) score={} nodos={}",
            if ok { "OK  " } else { "FAIL" }, nombre, uci, esperada, sc, nodes
        );
    }
    println!("{}", if todo_ok { "TODOS LOS MATES OK" } else { "HAY FALLOS EN LA SUITE DE MATES" });
}

fn run_prueba_apertura() {
    // 1.e4 d5 2.exd5 Qxd5 -- el motor deberia enrocar en unas pocas jugadas
    // propias en vez de salir a cazar peones con la dama (bug historico de v1).
    let mut b = Board::from_fen(STARTPOS).unwrap();
    let mut s = Searcher::new(64);
    let jugadas_iniciales = ["e2e4", "d7d5", "e4d5", "d8d5"];
    for uci in jugadas_iniciales {
        b = aplicar_uci(&b, uci);
    }
    println!("Posicion tras 1.e4 d5 2.exd5 Qxd5, buscando 10 jugadas propias del motor...");
    let mut enrocó = false;
    for i in 0..10 {
        let (mv, sc, nodes) = s.search_fixed_depth(&b, 5);
        let mv = match mv {
            Some(m) => m,
            None => break,
        };
        println!("  jugada propia #{}: {} (score {}, {} nodos)", i + 1, mv.to_uci(), sc, nodes);
        if mv.flag == MoveFlag::CastleKing || mv.flag == MoveFlag::CastleQueen {
            enrocó = true;
        }
        b = b.make_move(&mv);
        if b.turn == types::Color::Black {
            // saltar la respuesta rival con una jugada legal cualquiera (la primera)
            let respuestas = movegen::generate_legal(&b);
            if let Some(r) = respuestas.first() {
                b = b.make_move(r);
            }
        }
        if enrocó {
            break;
        }
    }
    println!("{}", if enrocó { "OK: el motor enrocó" } else { "el motor NO enrocó en 10 jugadas" });
}

fn aplicar_uci(b: &Board, uci: &str) -> Board {
    let mv = parse_uci_move(b, uci).expect("jugada UCI invalida");
    b.make_move(&mv)
}

fn parse_uci_move(b: &Board, uci: &str) -> Option<Move> {
    let moves = movegen::generate_legal(b);
    let bytes = uci.as_bytes();
    if bytes.len() < 4 {
        return None;
    }
    let from = types::square_from_name(&uci[0..2])?;
    let to = types::square_from_name(&uci[2..4])?;
    let promo = if bytes.len() >= 5 { PieceType::from_char(bytes[4] as char) } else { None };
    moves.into_iter().find(|m| m.from == from && m.to == to && m.promotion == promo)
}

/// Construye una jugada SIN validar legalidad real de movimiento (solo
/// determina el flag -- Captura/AlPaso/Silenciosa -- a partir del estado
/// del tablero). Necesario para los casos sinteticos de test de SEE, donde
/// el "atacante inicial" puede estar bloqueado en la realidad (igual que
/// hace el test_see.py de Python, que tampoco valida legalidad del primer
/// movimiento -- SEE en si mismo confia en que el llamador ya eligio un
/// atacante valido).
fn jugada_sintetica(b: &Board, uci: &str) -> Move {
    let from = types::square_from_name(&uci[0..2]).unwrap();
    let to = types::square_from_name(&uci[2..4]).unwrap();
    let es_al_paso = b.piece_at(from).map(|(_, pt)| pt) == Some(PieceType::Pawn) && Some(to) == b.ep_square;
    let flag = if es_al_paso {
        MoveFlag::EnPassant
    } else if b.piece_at(to).is_some() {
        MoveFlag::Capture
    } else {
        MoveFlag::Quiet
    };
    Move::new(from, to, None, flag)
}

fn run_cxb4_bug() {
    let fen = "r1b1k2r/ppqp1ppp/4p3/4n3/1b6/2PQBN2/P1P2PPP/R3KB1R w KQkq - 0 11";
    let b = Board::from_fen(fen).unwrap();
    let mut s = Searcher::new(64);
    let (mv, sc, nodes) = s.search_fixed_depth(&b, 6);
    let uci = mv.map(|m| m.to_uci()).unwrap_or_default();
    println!("FEN bug cxb4: jugada elegida = {} (score {}, {} nodos)", uci, sc, nodes);
    println!("{}", if uci == "c3b4" { "eligio cxb4 (igual que v3 sin proteccion)" } else { "NO eligio cxb4" });
}

fn run_see_tests() {
    let mut ok = true;
    let casos: Vec<(&str, &str, &str, i32)> = vec![
        ("Caso 1 (captura libre)", "k7/8/8/7p/8/8/8/K6R w - - 0 1", "h1h5", 100),
        ("Caso 2 (1v1, mal cambio)", "k2r4/8/8/3p4/8/8/8/K2R4 w - - 0 1", "d1d5", 100 - 500),
        ("Caso 3 (2 atacantes vs 1 defensor)", "k2r4/8/8/3p4/8/8/3R4/K2R4 w - - 0 1", "d1d5", 100),
        ("Caso 4 (cxd5 con recaptura de peon)", "k7/8/4p3/3n4/2P5/1N6/8/K7 w - - 0 1", "c4d5", 220),
        ("Caso 5 (al paso, sin recaptura)", "k7/8/8/3pP3/8/8/8/K7 w - d6 0 1", "e5d6", 100),
        ("Caso 5b (al paso con recaptura)", "k7/2p5/8/3pP3/8/8/8/K7 w - d6 0 1", "e5d6", 0),
        ("Caso 6 (FEN del bug, cxb4)", "r1b1k2r/ppqp1ppp/4p3/4n3/1b6/2PQBN2/P1P2PPP/R3KB1R w KQkq - 0 11", "c3b4", 330),
    ];
    for (nombre, fen, uci, esperado) in &casos {
        let b = Board::from_fen(fen).unwrap();
        let mv = jugada_sintetica(&b, uci);
        let r = see::see(&b, &mv);
        let pass = r == *esperado;
        ok &= pass;
        println!("{} {}: see={} esperado={}", if pass { "OK  " } else { "FALLO" }, nombre, r, esperado);
    }
    println!("{}", if ok { "TODOS LOS CASOS DE SEE OK" } else { "HAY FALLOS EN SEE" });

    // Oraculo de fuerza bruta: posiciones al azar (self-play con jugadas
    // aleatorias desde la posicion inicial), comparando see() contra un
    // minimax real sobre TODAS las jugadas de captura posibles.
    println!("\nVerificando contra oraculo de fuerza bruta...");
    let mut rng_state: u64 = 0xC0FFEE1234567u64;
    let mut next_rand = move || {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        rng_state
    };

    let mut total_capturas = 0u64;
    let mut discrepancias = 0u64;
    for _partida in 0..5000 {
        let mut b = Board::startpos();
        let plies = 4 + (next_rand() % 20) as u32;
        let mut valida = true;
        for _ in 0..plies {
            let moves = movegen::generate_legal(&b);
            if moves.is_empty() {
                valida = false;
                break;
            }
            let mv = moves[(next_rand() as usize) % moves.len()];
            b = b.make_move(&mv);
        }
        if !valida {
            continue;
        }
        let capturas: Vec<Move> = movegen::generate_legal(&b).into_iter().filter(|m| m.is_capture()).collect();
        for mv in capturas {
            let rapido = see::see(&b, &mv);
            let oraculo = see::see_oracle(&b, &mv);
            total_capturas += 1;
            if rapido != oraculo {
                discrepancias += 1;
                println!(
                    "  DISCREPANCIA: fen='{}' jugada={} see={} oraculo={}",
                    b.to_fen(), mv.to_uci(), rapido, oraculo
                );
            }
        }
    }
    println!(
        "{} capturas comparadas, {} discrepancias -- {}",
        total_capturas, discrepancias,
        if discrepancias == 0 { "SEE COINCIDE CON EL ORACULO" } else { "HAY DISCREPANCIAS, revisar" }
    );
}

fn uci_loop() {
    let stdin = io::stdin();
    let mut board = Board::from_fen(STARTPOS).unwrap();
    let mut searcher = Searcher::new(64);

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let partes: Vec<&str> = line.split_whitespace().collect();
        match partes[0] {
            "uci" => {
                println!("id name MiMotor Tal v6 (Rust)");
                println!("id author Tavito y Claude");
                println!("option name Hash type spin default 64 min 1 max 1024");
                println!("uciok");
                io::stdout().flush().ok();
            }
            "isready" => {
                println!("readyok");
                io::stdout().flush().ok();
            }
            "ucinewgame" => {
                board = Board::from_fen(STARTPOS).unwrap();
                searcher.clear_tt();
            }
            "position" => {
                let mut idx = 1;
                if partes.get(1) == Some(&"startpos") {
                    board = Board::from_fen(STARTPOS).unwrap();
                    idx = 2;
                } else if partes.get(1) == Some(&"fen") {
                    let moves_pos = partes.iter().position(|&p| p == "moves").unwrap_or(partes.len());
                    let fen = partes[2..moves_pos].join(" ");
                    board = match Board::from_fen(&fen) {
                        Ok(b) => b,
                        Err(e) => {
                            println!("info string error de FEN: {}", e);
                            continue;
                        }
                    };
                    idx = moves_pos;
                }
                if partes.get(idx) == Some(&"moves") {
                    for uci in &partes[idx + 1..] {
                        if let Some(mv) = parse_uci_move(&board, uci) {
                            board = board.make_move(&mv);
                        }
                    }
                }
            }
            "go" => {
                let mut movetime: u64 = 2000;
                if let Some(i) = partes.iter().position(|&p| p == "movetime") {
                    movetime = partes.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(2000);
                } else if let Some(i) = partes.iter().position(|&p| p == "depth") {
                    let depth: i32 = partes.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(6);
                    let (mv, sc, _) = searcher.search_fixed_depth(&board, depth);
                    println!("info score cp {}", sc);
                    println!("bestmove {}", mv.map(|m| m.to_uci()).unwrap_or_else(|| "0000".to_string()));
                    io::stdout().flush().ok();
                    continue;
                } else if let Some(i) = partes.iter().position(|&p| p == "wtime") {
                    let wtime: i64 = partes.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(10000);
                    let btime_i = partes.iter().position(|&p| p == "btime");
                    let btime: i64 = btime_i.and_then(|j| partes.get(j + 1)).and_then(|s| s.parse().ok()).unwrap_or(10000);
                    let mio = if board.turn == types::Color::White { wtime } else { btime };
                    movetime = ((mio / 30).max(50)) as u64;
                }
                let (mv, sc) = searcher.search_time(&board, movetime, 64, |depth, score, nodes, ms| {
                    println!("info depth {} score cp {} nodes {} time {}", depth, score, nodes, ms);
                    io::stdout().flush().ok();
                });
                let _ = sc;
                println!("bestmove {}", mv.map(|m| m.to_uci()).unwrap_or_else(|| "0000".to_string()));
                io::stdout().flush().ok();
            }
            "quit" => break,
            _ => {}
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 {
        match args[1].as_str() {
            "perft" => {
                run_perft_suite();
                return;
            }
            "divide" if args.len() > 3 => {
                let depth: u32 = args[2].parse().expect("profundidad invalida");
                let fen = args[3..].join(" ");
                run_divide(&fen, depth);
                return;
            }
            "bench" => {
                let depth: i32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(6);
                run_bench(depth);
                return;
            }
            "matetest" => {
                run_mate_tests();
                return;
            }
            "aperturatest" => {
                run_prueba_apertura();
                return;
            }
            "cxb4test" => {
                run_cxb4_bug();
                return;
            }
            "seetest" => {
                run_see_tests();
                return;
            }
            _ => {}
        }
    }
    uci_loop();
}
