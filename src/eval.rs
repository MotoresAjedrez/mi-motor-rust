// Evaluación estática, portada de mi_motor.py (identidad Tal) -- mismos
// valores numéricos ya ajustados en Python, no reinventados. Devuelve el
// puntaje en centipeones desde el punto de vista del bando que mueve.

use crate::bitboard::{
    bishop_attacks, king_attacks, knight_attacks, pawn_attacks, popcount, queen_attacks,
    rook_attacks, Bitboard,
};
use crate::board::Board;
use crate::types::{file_of, make_square, rank_of, Color, PieceType, Square};
use std::sync::atomic::{AtomicU8, Ordering};

// Dos identidades de evaluacion, seleccionables en caliente (UCI "setoption
// name Personalidad" o variable de entorno MIMOTOR_PERSONALIDAD), que
// COEXISTEN -- no se reemplaza Tal, se agrega Universal como alternativa.
// Estado global de solo-lectura durante la busqueda (se fija antes de "go",
// nunca cambia a mitad de una busqueda concurrente) -- por eso un atomico
// simple alcanza, sin necesitar pasar el parametro por cada llamada interna.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Personalidad {
    Tal,
    Universal,
}

static PERSONALIDAD_ACTUAL: AtomicU8 = AtomicU8::new(0);

pub fn set_personalidad(p: Personalidad) {
    PERSONALIDAD_ACTUAL.store(if p == Personalidad::Universal { 1 } else { 0 }, Ordering::Relaxed);
}

pub fn personalidad_desde_texto(s: &str) -> Option<Personalidad> {
    match s.to_lowercase().as_str() {
        "tal" => Some(Personalidad::Tal),
        "universal" => Some(Personalidad::Universal),
        _ => None,
    }
}

fn personalidad_actual() -> Personalidad {
    if PERSONALIDAD_ACTUAL.load(Ordering::Relaxed) == 1 { Personalidad::Universal } else { Personalidad::Tal }
}

const VALOR: [i32; 6] = [100, 320, 330, 500, 900, 0]; // Pawn,Knight,Bishop,Rook,Queen,King
const ESCALA_MATERIAL_TAL: f64 = 0.9;
const ESCALA_MATERIAL_UNIVERSAL: f64 = 1.0;
const TEMPO: i32 = 12;

const PESO_MOV: [i32; 6] = [0, 4, 4, 3, 1, 0];
const PESO_ATQ: [i32; 6] = [0, 2, 2, 3, 5, 0];
// v12: bajado de 1.45 a 1.15. Diagnostico de divergencia contra Stockfish
// (profundidad 18) en las posiciones criticas de 2 derrotas reales contra
// simpleEval mostro a Tal sobreestimando la posicion propia por 100-380cp de
// forma sostenida -- comparado en la MISMA posicion, Universal (factor 1.0)
// daba un numero mucho mas cercano a Stockfish. 1.45 inflaba demasiado el
// termino de ataque al rey (hasta +225cp extra en zonas muy atacadas), dando
// confianza excesiva en complicaciones sin comprobar si de verdad progresan.
// 1.15 mantiene la identidad agresiva (sigue por encima del 1.0 neutral de
// Universal) sin el sesgo tan marcado. Verificado con torneo h2h antes de
// aceptarlo (ver resultados_tal_calibrado_h2h.txt).
const FACTOR_ATAQUE_TAL: f64 = 1.15;
const FACTOR_ATAQUE_UNIVERSAL: f64 = 1.0; // ataque al rey de peso normal, sin el bono no-lineal agresivo de Tal

const TABLA_SEGURIDAD: [i32; 62] = [
    0, 0, 1, 2, 3, 5, 7, 9, 12, 15, 18, 22, 26, 30, 35, 39, 44, 50, 56, 62, 68, 75, 82, 85, 89,
    97, 105, 113, 122, 131, 140, 150, 169, 180, 191, 202, 213, 225, 237, 248, 260, 272, 283, 295,
    307, 318, 330, 342, 354, 366, 377, 389, 401, 412, 424, 436, 448, 459, 471, 483, 494, 500,
];

// Piece-square tables, escritas visualmente (primera fila = 8a fila = rank 7 en indice 0..8).
#[rustfmt::skip]
const PEON_MG: [i32; 64] = [
      0,   0,   0,   0,   0,   0,   0,   0,
     50,  50,  50,  50,  50,  50,  50,  50,
     10,  10,  20,  30,  30,  20,  10,  10,
      5,   5,  10,  25,  25,  10,   5,   5,
      0,   0,   0,  20,  20,   0,   0,   0,
      5,  -5, -10,   0,   0, -10,  -5,   5,
      5,  10,  10, -20, -20,  10,  10,   5,
      0,   0,   0,   0,   0,   0,   0,   0,
];
#[rustfmt::skip]
const PEON_EG: [i32; 64] = [
      0,   0,   0,   0,   0,   0,   0,   0,
     80,  80,  80,  80,  80,  80,  80,  80,
     50,  50,  50,  50,  50,  50,  50,  50,
     30,  30,  30,  30,  30,  30,  30,  30,
     15,  15,  15,  15,  15,  15,  15,  15,
      5,   5,   5,   5,   5,   5,   5,   5,
      0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,
];
#[rustfmt::skip]
const CABALLO: [i32; 64] = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
];
#[rustfmt::skip]
const ALFIL: [i32; 64] = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
];
#[rustfmt::skip]
const TORRE: [i32; 64] = [
      0,   0,   0,   0,   0,   0,   0,   0,
      5,  10,  10,  10,  10,  10,  10,   5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
      0,   0,   0,   5,   5,   0,   0,   0,
];
#[rustfmt::skip]
const DAMA: [i32; 64] = [
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,   5,   5,   5,   0, -10,
     -5,   0,   5,   5,   5,   5,   0,  -5,
      0,   0,   5,   5,   5,   5,   0,  -5,
    -10,   5,   5,   5,   5,   5,   0, -10,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20,
];
#[rustfmt::skip]
const REY_MG: [i32; 64] = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
     20,  20,   0,   0,   0,   0,  20,  20,
     20,  30,  10,   0,   0,  10,  30,  20,
];
#[rustfmt::skip]
const REY_EG: [i32; 64] = [
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10,   0,   0, -10, -20, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -30,   0,   0,   0,   0, -30, -30,
    -50, -30, -30, -30, -30, -30, -30, -50,
];

fn pst_mg(pt: PieceType) -> &'static [i32; 64] {
    match pt {
        PieceType::Pawn => &PEON_MG,
        PieceType::Knight => &CABALLO,
        PieceType::Bishop => &ALFIL,
        PieceType::Rook => &TORRE,
        PieceType::Queen => &DAMA,
        PieceType::King => &REY_MG,
    }
}

fn pst_eg(pt: PieceType) -> &'static [i32; 64] {
    match pt {
        PieceType::Pawn => &PEON_EG,
        PieceType::Knight => &CABALLO,
        PieceType::Bishop => &ALFIL,
        PieceType::Rook => &TORRE,
        PieceType::Queen => &DAMA,
        PieceType::King => &REY_EG,
    }
}

fn piece_attacks(pt: PieceType, sq: Square, occ: Bitboard, color: Color) -> Bitboard {
    match pt {
        PieceType::Pawn => pawn_attacks(color, sq),
        PieceType::Knight => knight_attacks(sq),
        PieceType::Bishop => bishop_attacks(sq, occ),
        PieceType::Rook => rook_attacks(sq, occ),
        PieceType::Queen => queen_attacks(sq, occ),
        PieceType::King => king_attacks(sq),
    }
}

fn distancia_chebyshev(a: Square, b: Square) -> i32 {
    let (fa, ra) = (file_of(a) as i32, rank_of(a) as i32);
    let (fb, rb) = (file_of(b) as i32, rank_of(b) as i32);
    (fa - fb).abs().max((ra - rb).abs())
}

// Indice de avance hacia la coronacion (0 = fila de salida, 7 = coronando),
// independiente del color: para blancas es directamente rank_of, para negras
// es la fila espejada.
fn indice_avance(color: Color, sq: Square) -> usize {
    let r = rank_of(sq) as usize;
    if color == Color::White { r } else { 7 - r }
}

// Peon pasado: ningun peon rival en la misma columna ni en las adyacentes,
// por delante de esta casilla en direccion a la coronacion. Se calcula con
// mascaras de fila/columna en vez de generar el tablero completo -- barato,
// se llama una vez por peon en cada evaluacion.
fn es_pasado(color: Color, sq: Square, peones_rivales: Bitboard) -> bool {
    let f = file_of(sq) as i32;
    let r = rank_of(sq) as i32;
    let mut columnas: Bitboard = 0;
    for df in -1..=1i32 {
        let nf = f + df;
        if (0..8).contains(&nf) {
            columnas |= 0x0101010101010101u64 << nf;
        }
    }
    let filas_delante: Bitboard = if color == Color::White {
        if r == 7 { 0 } else { !0u64 << ((r + 1) * 8) }
    } else if r == 0 {
        0
    } else {
        !(!0u64 << (r * 8))
    };
    (peones_rivales & columnas & filas_delante) == 0
}

const PASO_BONUS: [i32; 8] = [0, 5, 10, 20, 35, 60, 100, 150];
const VENTAJA_DECISIVA: f64 = 500.0; // ~una torre de diferencia
const PESO_ACERCAMIENTO_REY: f64 = 4.0;

// Puesto avanzado (profilaxis, solo personalidad Universal): pieza menor
// propia en territorio rival que ningun peon rival puede llegar a atacar
// nunca (mismo patron de mascara que un peon pasado, pero mirando si HAY
// peones rivales adelante en columnas adyacentes, no si NO los hay). Premia
// restringir el territorio del rival, no solo maximizar la movilidad propia
// -- una pieza asi no se puede expulsar con un peon, limita permanentemente
// los planes del rival en esa zona del tablero.
fn en_territorio_rival(color: Color, sq: Square) -> bool {
    let r = rank_of(sq);
    if color == Color::White { r >= 4 } else { r <= 3 }
}

fn puesto_avanzado_seguro(color: Color, sq: Square, peones_rivales: Bitboard) -> bool {
    es_pasado(color, sq, peones_rivales) // misma mascara: sin peon rival por delante en columnas adyacentes
}

const BONUS_PUESTO: [i32; 6] = [0, 25, 12, 0, 0, 0]; // solo caballo/alfil

fn pareja_alfiles_bonus(color: Color, b: &Board) -> f64 {
    if popcount(b.pieces[color as usize][PieceType::Bishop as usize]) >= 2 { 35.0 } else { 0.0 }
}

fn king_zone(ksq: Square, color: Color) -> Bitboard {
    let mut z = king_attacks(ksq) | (1u64 << ksq);
    let f = file_of(ksq) as i32;
    let r = rank_of(ksq) as i32 + if color == Color::White { 2 } else { -2 };
    if (0..8).contains(&r) {
        for df in -1..=1i32 {
            let nf = f + df;
            if (0..8).contains(&nf) {
                z |= 1u64 << make_square(nf as u8, r as u8);
            }
        }
    }
    z
}

struct AtaqueRey {
    ataque_w: f64,
    ataque_b: f64,
}

fn calcular_ataque_rey(b: &Board, factor_ataque: f64) -> AtaqueRey {
    let rey_w = b.king_square(Color::White);
    let rey_b = b.king_square(Color::Black);
    let zona_w = king_zone(rey_w, Color::White);
    let zona_b = king_zone(rey_b, Color::Black);

    let mut unidades = [0i32; 2]; // [negro, blanco] como en python (indice 0=negras,1=blancas)
    let mut n_atacantes = [0i32; 2];

    for (color, idx, za) in [(Color::White, 1usize, zona_b), (Color::Black, 0usize, zona_w)] {
        let occ_pieces = b.pieces[color as usize];
        let mut pawns = occ_pieces[PieceType::Pawn as usize];
        while pawns != 0 {
            let sq = crate::bitboard::pop_lsb(&mut pawns);
            let u = popcount(pawn_attacks(color, sq) & za) as i32;
            unidades[idx] += u;
        }
        for pt in [PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen] {
            let mut bb = occ_pieces[pt as usize];
            while bb != 0 {
                let sq = crate::bitboard::pop_lsb(&mut bb);
                let att = piece_attacks(pt, sq, b.occupied, color);
                let u = popcount(att & za) as i32;
                if u > 0 {
                    unidades[idx] += PESO_ATQ[pt as usize] * u;
                    n_atacantes[idx] += 1;
                }
            }
        }
    }

    let puntaje = |idx: usize, color: Color| -> f64 {
        let mut u = unidades[idx];
        if u <= 0 {
            return 0.0;
        }
        let tiene_dama = b.pieces[color as usize][PieceType::Queen as usize] != 0;
        if !tiene_dama {
            u /= 2;
        }
        let mut s = TABLA_SEGURIDAD[u.min(61) as usize] as f64 * factor_ataque;
        if n_atacantes[idx] < 2 {
            s *= 0.35;
        }
        let rey_rival = if color == Color::White { rey_b } else { rey_w };
        let frey = file_of(rey_rival);
        if frey <= 2 || frey >= 6 {
            s *= 1.15;
        }
        s
    };

    AtaqueRey { ataque_w: puntaje(1, Color::White), ataque_b: puntaje(0, Color::Black) }
}

pub fn evaluate(b: &Board) -> i32 {
    let pers = personalidad_actual();
    let es_universal = pers == Personalidad::Universal;
    let escala_material = if es_universal { ESCALA_MATERIAL_UNIVERSAL } else { ESCALA_MATERIAL_TAL };
    let factor_ataque = if es_universal { FACTOR_ATAQUE_UNIVERSAL } else { FACTOR_ATAQUE_TAL };
    // Universal castiga mas la estructura de peones rota (filosofia tecnica
    // clasica); Tal la castiga poco a proposito, para no desalentar sacrificios.
    let peso_estructura: i32 = if es_universal { 16 } else { 8 };
    // Universal se apoya mas en la tecnica de finales (rey activo, peones
    // pasados, torres activas) ya que esa es literalmente su identidad.
    let peso_final_extra: f64 = if es_universal { 1.3 } else { 1.0 };

    let occ_w = b.occupied_co[Color::White as usize];
    let occ_b = b.occupied_co[Color::Black as usize];

    let fase = (popcount(b.pieces[0][PieceType::Knight as usize] | b.pieces[1][PieceType::Knight as usize])
        + popcount(b.pieces[0][PieceType::Bishop as usize] | b.pieces[1][PieceType::Bishop as usize])
        + 2 * popcount(b.pieces[0][PieceType::Rook as usize] | b.pieces[1][PieceType::Rook as usize])
        + 4 * popcount(b.pieces[0][PieceType::Queen as usize] | b.pieces[1][PieceType::Queen as usize]))
        .min(24) as f64;
    let mgf = fase / 24.0;
    let egf = 1.0 - mgf;

    let mut mat = [0i32; 2];
    let mut pst_mg_sum = [0i32; 2];
    let mut pst_eg_sum = [0i32; 2];
    let mut movilidad = [0i32; 2];

    for pt in crate::types::ALL_PIECE_TYPES {
        let vpt = VALOR[pt as usize];
        let m = pst_mg(pt);
        let e = pst_eg(pt);
        for (color, idx) in [(Color::White, 1usize), (Color::Black, 0usize)] {
            let mut bb = b.pieces[color as usize][pt as usize];
            while bb != 0 {
                let sq = crate::bitboard::pop_lsb(&mut bb);
                mat[idx] += vpt;
                let pst_idx = if color == Color::White { (sq ^ 56) as usize } else { sq as usize };
                pst_mg_sum[idx] += m[pst_idx];
                pst_eg_sum[idx] += e[pst_idx];
                let peso = PESO_MOV[pt as usize];
                if peso != 0 {
                    let occ = b.occupied;
                    movilidad[idx] += peso * popcount(piece_attacks(pt, sq, occ, color)) as i32;
                }
            }
        }
    }

    // Estructura de peones: doblados y aislados (penalización deliberadamente baja)
    let pw = b.pieces[Color::White as usize][PieceType::Pawn as usize];
    let pb = b.pieces[Color::Black as usize][PieceType::Pawn as usize];
    let mut estructura = [0i32; 2];
    for f in 0..8u8 {
        let fm: Bitboard = 0x0101010101010101u64 << f;
        let adyacentes: Bitboard = (if f > 0 { 0x0101010101010101u64 << (f - 1) } else { 0 })
            | (if f < 7 { 0x0101010101010101u64 << (f + 1) } else { 0 });
        let cw = popcount(pw & fm) as i32;
        let cb = popcount(pb & fm) as i32;
        if cw > 0 {
            if cw > 1 {
                estructura[1] -= peso_estructura * (cw - 1);
            }
            if pw & adyacentes == 0 {
                estructura[1] -= peso_estructura * cw;
            }
        }
        if cb > 0 {
            if cb > 1 {
                estructura[0] -= peso_estructura * (cb - 1);
            }
            if pb & adyacentes == 0 {
                estructura[0] -= peso_estructura * cb;
            }
        }
    }

    // Escudo de peones frente al propio rey (solo pesa en medio juego)
    let escudo = |ksq: Square, color: Color, propios: Bitboard| -> i32 {
        let mut s = 0;
        let f = file_of(ksq) as i32;
        let r = rank_of(ksq) as i32;
        let dr = if color == Color::White { 1 } else { -1 };
        for df in -1..=1i32 {
            let nf = f + df;
            if (0..8).contains(&nf) {
                let r1 = r + dr;
                let r2 = r + 2 * dr;
                if (0..8).contains(&r1) && propios & (1u64 << make_square(nf as u8, r1 as u8)) != 0 {
                    s += 12;
                } else if (0..8).contains(&r2) && propios & (1u64 << make_square(nf as u8, r2 as u8)) != 0 {
                    s += 6;
                }
            }
        }
        s
    };
    let rey_w = b.king_square(Color::White);
    let rey_b = b.king_square(Color::Black);
    let escudo_w = escudo(rey_w, Color::White, pw) as f64 * mgf;
    let escudo_b = escudo(rey_b, Color::Black, pb) as f64 * mgf;

    let ar = calcular_ataque_rey(b, factor_ataque);

    // Peones pasados: bono creciente por avance, reforzado por escolta del
    // rey propio (mas cerca de la casilla de coronacion que el rey rival) y
    // a plena fuerza en el final -- en medio juego pesa la mitad porque
    // todavia no es facil de convertir con piezas en el tablero.
    let mut pasados = [0.0f64; 2];
    for (color, idx, propios, rivales) in [(Color::White, 1usize, pw, pb), (Color::Black, 0usize, pb, pw)] {
        let mut bb = propios;
        while bb != 0 {
            let sq = crate::bitboard::pop_lsb(&mut bb);
            if es_pasado(color, sq, rivales) {
                let base = PASO_BONUS[indice_avance(color, sq)] as f64;
                let mut bono = base * (0.5 + 0.5 * egf);
                let casilla_coronacion = if color == Color::White {
                    make_square(file_of(sq), 7)
                } else {
                    make_square(file_of(sq), 0)
                };
                let (rey_propio, rey_rival) =
                    if color == Color::White { (rey_w, rey_b) } else { (rey_b, rey_w) };
                let dist_propio = distancia_chebyshev(rey_propio, casilla_coronacion);
                let dist_rival = distancia_chebyshev(rey_rival, casilla_coronacion);
                bono += (dist_rival - dist_propio) as f64 * 3.0 * egf;
                pasados[idx] += bono;
            }
        }
    }

    // Torres en la 7ma fila (2da del rival) y en columnas abiertas/semi-
    // abiertas: mucho mas activas ahi, sobre todo en finales de torres.
    let mut torres_activas = [0.0f64; 2];
    for (color, idx, propios, rivales) in [(Color::White, 1usize, pw, pb), (Color::Black, 0usize, pb, pw)] {
        let mut bb = b.pieces[color as usize][PieceType::Rook as usize];
        while bb != 0 {
            let sq = crate::bitboard::pop_lsb(&mut bb);
            let r = rank_of(sq);
            let en_7ma = if color == Color::White { r == 6 } else { r == 1 };
            if en_7ma {
                torres_activas[idx] += 20.0;
            }
            let f = file_of(sq);
            let columna: Bitboard = 0x0101010101010101u64 << f;
            let hay_propio = propios & columna != 0;
            let hay_rival = rivales & columna != 0;
            if !hay_propio && !hay_rival {
                torres_activas[idx] += 15.0;
            } else if !hay_propio {
                torres_activas[idx] += 8.0;
            }
        }
    }

    // Actividad de rey en finales con ventaja decisiva: cuando el material
    // ya alcanza para ganar (mas de una torre de diferencia), el objetivo
    // deja de ser "no arriesgar" y pasa a "progresar" -- se premia que el
    // rey del bando que gana se acerque al rey rival (tecnica clasica de
    // conversion). Solo pesa en el final (egf) y solo del lado que gana.
    let dif_material = (mat[1] - mat[0]) as f64 * escala_material;
    let cierre_rey = if dif_material > VENTAJA_DECISIVA {
        (7 - distancia_chebyshev(rey_w, rey_b)) as f64 * PESO_ACERCAMIENTO_REY * egf
    } else if dif_material < -VENTAJA_DECISIVA {
        -((7 - distancia_chebyshev(rey_w, rey_b)) as f64 * PESO_ACERCAMIENTO_REY * egf)
    } else {
        0.0
    };

    // Bloque especifico de personalidad: Tal mantiene piezas de ataque
    // cuando hay iniciativa (identidad agresiva/sacrificial); Universal en
    // cambio busca pareja de alfiles y restringe al rival con puestos
    // avanzados seguros (profilaxis).
    let mut extra = 0.0f64;
    if !es_universal {
        if ar.ataque_w - ar.ataque_b > 60.0 {
            if b.pieces[Color::White as usize][PieceType::Queen as usize] & occ_w != 0 {
                extra += 30.0;
            }
            extra += 8.0 * popcount(b.pieces[Color::White as usize][PieceType::Rook as usize] & occ_w).min(2) as f64;
        } else if ar.ataque_b - ar.ataque_w > 60.0 {
            if b.pieces[Color::Black as usize][PieceType::Queen as usize] & occ_b != 0 {
                extra -= 30.0;
            }
            extra -= 8.0 * popcount(b.pieces[Color::Black as usize][PieceType::Rook as usize] & occ_b).min(2) as f64;
        }
    } else {
        extra += pareja_alfiles_bonus(Color::White, b) - pareja_alfiles_bonus(Color::Black, b);

        for (color, signo, rivales) in [(Color::White, 1.0f64, pb), (Color::Black, -1.0f64, pw)] {
            for pt in [PieceType::Knight, PieceType::Bishop] {
                let mut bb = b.pieces[color as usize][pt as usize];
                while bb != 0 {
                    let sq = crate::bitboard::pop_lsb(&mut bb);
                    if en_territorio_rival(color, sq) && puesto_avanzado_seguro(color, sq, rivales) {
                        extra += signo * BONUS_PUESTO[pt as usize] as f64;
                    }
                }
            }
        }
    }

    // Simplificar hacia un final ganado -- o EVITAR simplificar estando peor
    // -- aplica a las DOS personalidades por igual, no es un gusto de estilo
    // sino un principio basico: cambiar piezas (sobre todo damas) estando
    // material abajo apaga las complicaciones/chances de swindle y hace
    // trivial la tecnica de conversion del rival. Antes vivia solo en el
    // bloque de Universal; se generaliza aca porque es correcto para
    // cualquier personalidad, no una cuestion de estilo.
    let piezas_no_peon = crate::types::ALL_PIECE_TYPES
        .iter()
        .filter(|&&pt| pt != PieceType::Pawn && pt != PieceType::King)
        .map(|&pt| popcount(b.pieces[0][pt as usize] | b.pieces[1][pt as usize]))
        .sum::<u32>() as f64;
    const UMBRAL_SIMPLIFICACION: f64 = 150.0;
    if dif_material > UMBRAL_SIMPLIFICACION {
        extra += (10.0 - piezas_no_peon) * 2.0 * (0.4 + 0.6 * egf);
    } else if dif_material < -UMBRAL_SIMPLIFICACION {
        extra -= (10.0 - piezas_no_peon) * 2.0 * (0.4 + 0.6 * egf);
    }

    let total = (mat[1] - mat[0]) as f64 * escala_material
        + (pst_mg_sum[1] - pst_mg_sum[0]) as f64 * mgf
        + (pst_eg_sum[1] - pst_eg_sum[0]) as f64 * egf
        + (movilidad[1] - movilidad[0]) as f64
        + (ar.ataque_w - ar.ataque_b)
        + (estructura[1] - estructura[0]) as f64
        + (escudo_w - escudo_b)
        + (pasados[1] - pasados[0]) * peso_final_extra
        + (torres_activas[1] - torres_activas[0]) * peso_final_extra
        + cierre_rey * peso_final_extra
        + extra;

    let total_i = total.round() as i32;
    let perspectiva = if b.turn == Color::White { total_i } else { -total_i };
    let clasica = perspectiva + TEMPO;

    // v13: correccion opcional de una red neuronal ligera (ver neural.rs),
    // APAGADA por defecto (UCI "UseNN"). Aditiva, no reemplaza la
    // evaluacion clasica -- eval_final = eval_clasica + peso_red*eval_red,
    // con peso_red chico a proposito: la red es una correccion, no un
    // reemplazo de algo ya probado en cientos de partidas. Si no esta
    // activada o no hay pesos cargados, eval_red() devuelve None y esto
    // es exactamente lo mismo que antes (cero costo, cero cambio).
    const PESO_RED: f64 = 0.2;
    match crate::neural::eval_red(b) {
        Some(red_cp) => clasica + (PESO_RED * red_cp as f64).round() as i32,
        None => clasica,
    }
}
