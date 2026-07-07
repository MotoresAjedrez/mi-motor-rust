// Bitboards de 64 bits y tablas de ataques precalculadas.
// Piezas deslizantes: ray-casting clásico (no magic bitboards) -- documentado
// así explícitamente por decisión de tiempo: es más simple de verificar
// correctamente con perft, al costo de menos nodos/s que magic bitboards.
// Si hace falta más velocidad más adelante, este es el punto a optimizar.

use crate::types::{file_of, make_square, rank_of, Square};
use std::sync::OnceLock;

pub type Bitboard = u64;

pub const EMPTY: Bitboard = 0;

#[inline(always)]
pub fn bit(sq: Square) -> Bitboard {
    1u64 << sq
}

#[inline(always)]
pub fn popcount(bb: Bitboard) -> u32 {
    bb.count_ones()
}

#[inline(always)]
pub fn lsb(bb: Bitboard) -> Square {
    bb.trailing_zeros() as Square
}

#[inline(always)]
pub fn msb(bb: Bitboard) -> Square {
    63 - bb.leading_zeros() as Square
}

#[inline(always)]
pub fn pop_lsb(bb: &mut Bitboard) -> Square {
    let s = lsb(*bb);
    *bb &= *bb - 1;
    s
}

// 8 direcciones para piezas deslizantes.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Dir {
    N,
    S,
    E,
    W,
    NE,
    NW,
    SE,
    SW,
}

pub const ROOK_DIRS: [Dir; 4] = [Dir::N, Dir::S, Dir::E, Dir::W];
pub const BISHOP_DIRS: [Dir; 4] = [Dir::NE, Dir::NW, Dir::SE, Dir::SW];

// Direcciones "positivas" (aumentan el índice de casilla): el primer bloqueo
// se busca con lsb. Las "negativas" usan msb.
fn is_positive(dir: Dir) -> bool {
    matches!(dir, Dir::N | Dir::E | Dir::NE | Dir::NW)
}

struct Tables {
    knight: [Bitboard; 64],
    king: [Bitboard; 64],
    pawn_attacks: [[Bitboard; 64]; 2], // [color][square]
    rays: [[Bitboard; 64]; 8],         // [dir][square]
}

static TABLES: OnceLock<Tables> = OnceLock::new();

fn dir_index(dir: Dir) -> usize {
    match dir {
        Dir::N => 0,
        Dir::S => 1,
        Dir::E => 2,
        Dir::W => 3,
        Dir::NE => 4,
        Dir::NW => 5,
        Dir::SE => 6,
        Dir::SW => 7,
    }
}

fn step(file: i32, rank: i32, dir: Dir) -> Option<(i32, i32)> {
    let (df, dr) = match dir {
        Dir::N => (0, 1),
        Dir::S => (0, -1),
        Dir::E => (1, 0),
        Dir::W => (-1, 0),
        Dir::NE => (1, 1),
        Dir::NW => (-1, 1),
        Dir::SE => (1, -1),
        Dir::SW => (-1, -1),
    };
    let (nf, nr) = (file + df, rank + dr);
    if (0..8).contains(&nf) && (0..8).contains(&nr) {
        Some((nf, nr))
    } else {
        None
    }
}

fn build_tables() -> Tables {
    let mut knight = [0u64; 64];
    let mut king = [0u64; 64];
    let mut pawn_attacks = [[0u64; 64]; 2];
    let mut rays = [[0u64; 64]; 8];

    for sq in 0..64u8 {
        let f = file_of(sq) as i32;
        let r = rank_of(sq) as i32;

        // Caballo
        let knight_deltas = [
            (1, 2), (2, 1), (2, -1), (1, -2),
            (-1, -2), (-2, -1), (-2, 1), (-1, 2),
        ];
        for (df, dr) in knight_deltas {
            let (nf, nr) = (f + df, r + dr);
            if (0..8).contains(&nf) && (0..8).contains(&nr) {
                knight[sq as usize] |= bit(make_square(nf as u8, nr as u8));
            }
        }

        // Rey
        for df in -1..=1i32 {
            for dr in -1..=1i32 {
                if df == 0 && dr == 0 {
                    continue;
                }
                let (nf, nr) = (f + df, r + dr);
                if (0..8).contains(&nf) && (0..8).contains(&nr) {
                    king[sq as usize] |= bit(make_square(nf as u8, nr as u8));
                }
            }
        }

        // Peones (ataques diagonales, no incluye avance)
        for (color_idx, dr) in [(0usize, 1i32), (1usize, -1i32)] {
            for df in [-1i32, 1i32] {
                let (nf, nr) = (f + df, r + dr);
                if (0..8).contains(&nf) && (0..8).contains(&nr) {
                    pawn_attacks[color_idx][sq as usize] |= bit(make_square(nf as u8, nr as u8));
                }
            }
        }

        // Rayos para piezas deslizantes
        for &dir in ROOK_DIRS.iter().chain(BISHOP_DIRS.iter()) {
            let mut ray = 0u64;
            let (mut cf, mut cr) = (f, r);
            while let Some((nf, nr)) = step(cf, cr, dir) {
                ray |= bit(make_square(nf as u8, nr as u8));
                cf = nf;
                cr = nr;
            }
            rays[dir_index(dir)][sq as usize] = ray;
        }
    }

    Tables { knight, king, pawn_attacks, rays }
}

fn tables() -> &'static Tables {
    TABLES.get_or_init(build_tables)
}

pub fn knight_attacks(sq: Square) -> Bitboard {
    tables().knight[sq as usize]
}

pub fn king_attacks(sq: Square) -> Bitboard {
    tables().king[sq as usize]
}

pub fn pawn_attacks(color: crate::types::Color, sq: Square) -> Bitboard {
    tables().pawn_attacks[color as usize][sq as usize]
}

fn ray(dir: Dir, sq: Square) -> Bitboard {
    tables().rays[dir_index(dir)][sq as usize]
}

/// Ataques de una pieza deslizante en una dirección dada, respetando bloqueos.
fn sliding_attacks_dir(sq: Square, occupied: Bitboard, dir: Dir) -> Bitboard {
    let full_ray = ray(dir, sq);
    let blockers = full_ray & occupied;
    if blockers == 0 {
        return full_ray;
    }
    let blocker_sq = if is_positive(dir) { lsb(blockers) } else { msb(blockers) };
    full_ray & !ray(dir, blocker_sq)
}

pub fn rook_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    let mut attacks = 0u64;
    for &dir in ROOK_DIRS.iter() {
        attacks |= sliding_attacks_dir(sq, occupied, dir);
    }
    attacks
}

pub fn bishop_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    let mut attacks = 0u64;
    for &dir in BISHOP_DIRS.iter() {
        attacks |= sliding_attacks_dir(sq, occupied, dir);
    }
    attacks
}

pub fn queen_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    rook_attacks(sq, occupied) | bishop_attacks(sq, occupied)
}
