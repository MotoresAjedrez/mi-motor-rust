use crate::bitboard::{
    bishop_attacks, bit, king_attacks, knight_attacks, pawn_attacks, pop_lsb, queen_attacks,
    rook_attacks, Bitboard, EMPTY,
};
use crate::board::{Board, CASTLE_BK, CASTLE_BQ, CASTLE_WK, CASTLE_WQ};
use crate::types::{file_of, make_square, rank_of, Color, Move, MoveFlag, PieceType, Square};

pub type MoveList = Vec<Move>;

const PROMO_PIECES: [PieceType; 4] =
    [PieceType::Queen, PieceType::Rook, PieceType::Bishop, PieceType::Knight];

/// Genera todas las jugadas pseudo-legales (no filtra jaques propios).
pub fn generate_pseudo_legal(b: &Board) -> MoveList {
    let mut moves = Vec::with_capacity(48);
    let us = b.turn;
    let them = us.opposite();
    let own = b.occupied_co[us as usize];
    let enemy = b.occupied_co[them as usize];
    let occ = b.occupied;

    gen_pawn_moves(b, us, enemy, occ, &mut moves);
    gen_piece_moves(b, us, PieceType::Knight, own, occ, &mut moves, |sq, _| knight_attacks(sq));
    gen_piece_moves(b, us, PieceType::Bishop, own, occ, &mut moves, |sq, occ| bishop_attacks(sq, occ));
    gen_piece_moves(b, us, PieceType::Rook, own, occ, &mut moves, |sq, occ| rook_attacks(sq, occ));
    gen_piece_moves(b, us, PieceType::Queen, own, occ, &mut moves, |sq, occ| queen_attacks(sq, occ));
    gen_piece_moves(b, us, PieceType::King, own, occ, &mut moves, |sq, _| king_attacks(sq));
    gen_castling(b, us, &mut moves);

    moves
}

fn gen_piece_moves<F>(
    b: &Board,
    us: Color,
    pt: PieceType,
    own: Bitboard,
    occ: Bitboard,
    moves: &mut MoveList,
    attacks_fn: F,
) where
    F: Fn(Square, Bitboard) -> Bitboard,
{
    let mut pieces = b.pieces[us as usize][pt as usize];
    while pieces != 0 {
        let from = pop_lsb(&mut pieces);
        let mut targets = attacks_fn(from, occ) & !own;
        while targets != 0 {
            let to = pop_lsb(&mut targets);
            let flag = if bit(to) & occ != 0 { MoveFlag::Capture } else { MoveFlag::Quiet };
            moves.push(Move::new(from, to, None, flag));
        }
    }
}

fn gen_pawn_moves(b: &Board, us: Color, enemy: Bitboard, occ: Bitboard, moves: &mut MoveList) {
    let mut pawns = b.pieces[us as usize][PieceType::Pawn as usize];
    let (push_dir, start_rank, promo_rank): (i32, u8, u8) = match us {
        Color::White => (1, 1, 7),
        Color::Black => (-1, 6, 0),
    };

    while pawns != 0 {
        let from = pop_lsb(&mut pawns);
        let f = file_of(from) as i32;
        let r = rank_of(from) as i32;

        // Avance simple
        let one_rank = r + push_dir;
        if (0..8).contains(&one_rank) {
            let to = make_square(f as u8, one_rank as u8);
            if bit(to) & occ == 0 {
                push_pawn_move(moves, from, to, promo_rank == one_rank as u8, MoveFlag::Quiet);

                // Avance doble
                if rank_of(from) == start_rank {
                    let two_rank = r + 2 * push_dir;
                    let to2 = make_square(f as u8, two_rank as u8);
                    if bit(to2) & occ == 0 {
                        moves.push(Move::new(from, to2, None, MoveFlag::DoublePush));
                    }
                }
            }
        }

        // Capturas (incluye al paso)
        let mut att = pawn_attacks(us, from) & enemy;
        while att != 0 {
            let to = pop_lsb(&mut att);
            let is_promo = rank_of(to) == promo_rank;
            push_pawn_move(moves, from, to, is_promo, MoveFlag::Capture);
        }
        if let Some(ep) = b.ep_square {
            if pawn_attacks(us, from) & bit(ep) != 0 {
                moves.push(Move::new(from, ep, None, MoveFlag::EnPassant));
            }
        }
    }
}

fn push_pawn_move(moves: &mut MoveList, from: Square, to: Square, is_promo: bool, flag: MoveFlag) {
    if is_promo {
        for &p in PROMO_PIECES.iter() {
            moves.push(Move::new(from, to, Some(p), flag));
        }
    } else {
        moves.push(Move::new(from, to, None, flag));
    }
}

fn gen_castling(b: &Board, us: Color, moves: &mut MoveList) {
    let occ = b.occupied;
    match us {
        Color::White => {
            if b.castling_rights & CASTLE_WK != 0
                && occ & (bit(5) | bit(6)) == EMPTY
                && !b.is_square_attacked_by(4, Color::Black)
                && !b.is_square_attacked_by(5, Color::Black)
                && !b.is_square_attacked_by(6, Color::Black)
            {
                moves.push(Move::new(4, 6, None, MoveFlag::CastleKing));
            }
            if b.castling_rights & CASTLE_WQ != 0
                && occ & (bit(1) | bit(2) | bit(3)) == EMPTY
                && !b.is_square_attacked_by(4, Color::Black)
                && !b.is_square_attacked_by(3, Color::Black)
                && !b.is_square_attacked_by(2, Color::Black)
            {
                moves.push(Move::new(4, 2, None, MoveFlag::CastleQueen));
            }
        }
        Color::Black => {
            if b.castling_rights & CASTLE_BK != 0
                && occ & (bit(61) | bit(62)) == EMPTY
                && !b.is_square_attacked_by(60, Color::White)
                && !b.is_square_attacked_by(61, Color::White)
                && !b.is_square_attacked_by(62, Color::White)
            {
                moves.push(Move::new(60, 62, None, MoveFlag::CastleKing));
            }
            if b.castling_rights & CASTLE_BQ != 0
                && occ & (bit(57) | bit(58) | bit(59)) == EMPTY
                && !b.is_square_attacked_by(60, Color::White)
                && !b.is_square_attacked_by(59, Color::White)
                && !b.is_square_attacked_by(58, Color::White)
            {
                moves.push(Move::new(60, 58, None, MoveFlag::CastleQueen));
            }
        }
    }
}

/// Filtra las jugadas pseudo-legales: descarta las que dejan al propio rey en jaque.
pub fn generate_legal(b: &Board) -> MoveList {
    let us = b.turn;
    generate_pseudo_legal(b)
        .into_iter()
        .filter(|mv| {
            let after = b.make_move(mv);
            !after.in_check(us)
        })
        .collect()
}
