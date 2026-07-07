// Evaluación estática, portada de mi_motor.py (identidad Tal) -- mismos
// valores numéricos ya ajustados en Python, no reinventados. Devuelve el
// puntaje en centipeones desde el punto de vista del bando que mueve.

use crate::bitboard::{
    bishop_attacks, king_attacks, knight_attacks, pawn_attacks, popcount, queen_attacks,
    rook_attacks, Bitboard,
};
use crate::board::Board;
use crate::types::{file_of, make_square, rank_of, Color, PieceType, Square};

const VALOR: [i32; 6] = [100, 320, 330, 500, 900, 0]; // Pawn,Knight,Bishop,Rook,Queen,King
const ESCALA_MATERIAL: f64 = 0.9;
const TEMPO: i32 = 12;

const PESO_MOV: [i32; 6] = [0, 4, 4, 3, 1, 0];
const PESO_ATQ: [i32; 6] = [0, 2, 2, 3, 5, 0];
const FACTOR_ATAQUE: f64 = 1.45;

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

fn calcular_ataque_rey(b: &Board) -> AtaqueRey {
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
        let mut s = TABLA_SEGURIDAD[u.min(61) as usize] as f64 * FACTOR_ATAQUE;
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
                estructura[1] -= 8 * (cw - 1);
            }
            if pw & adyacentes == 0 {
                estructura[1] -= 8 * cw;
            }
        }
        if cb > 0 {
            if cb > 1 {
                estructura[0] -= 8 * (cb - 1);
            }
            if pb & adyacentes == 0 {
                estructura[0] -= 8 * cb;
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

    let ar = calcular_ataque_rey(b);

    // Mantener piezas de ataque cuando hay iniciativa
    let mut extra = 0i32;
    if ar.ataque_w - ar.ataque_b > 60.0 {
        if b.pieces[Color::White as usize][PieceType::Queen as usize] & occ_w != 0 {
            extra += 30;
        }
        extra += 8 * popcount(b.pieces[Color::White as usize][PieceType::Rook as usize] & occ_w).min(2) as i32;
    } else if ar.ataque_b - ar.ataque_w > 60.0 {
        if b.pieces[Color::Black as usize][PieceType::Queen as usize] & occ_b != 0 {
            extra -= 30;
        }
        extra -= 8 * popcount(b.pieces[Color::Black as usize][PieceType::Rook as usize] & occ_b).min(2) as i32;
    }

    let total = (mat[1] - mat[0]) as f64 * ESCALA_MATERIAL
        + (pst_mg_sum[1] - pst_mg_sum[0]) as f64 * mgf
        + (pst_eg_sum[1] - pst_eg_sum[0]) as f64 * egf
        + (movilidad[1] - movilidad[0]) as f64
        + (ar.ataque_w - ar.ataque_b)
        + (estructura[1] - estructura[0]) as f64
        + (escudo_w - escudo_b)
        + extra as f64;

    let total_i = total.round() as i32;
    let perspectiva = if b.turn == Color::White { total_i } else { -total_i };
    perspectiva + TEMPO
}
