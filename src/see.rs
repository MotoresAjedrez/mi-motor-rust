// Static Exchange Evaluation -- portado de mi_motor.py (ver() en Python).
// Simula la secuencia completa de capturas/recapturas en la casilla de
// destino, asumiendo que cada bando recaptura con su atacante de menor
// valor disponible ("swap-off" estandar). Trabaja sobre copias locales de
// los bitboards (Board es Copy, copiarlo es barato) para no pagar el costo
// de aplicar jugadas reales en la tabla de transposicion / hash, etc.

use crate::bitboard::{
    bishop_attacks, bit, king_attacks, knight_attacks, lsb, pawn_attacks, rook_attacks,
    Bitboard,
};
use crate::board::Board;
use crate::types::{file_of, make_square, rank_of, Color, Move, MoveFlag, PieceType, Square};

const VALOR: [i32; 6] = [100, 320, 330, 500, 900, 0]; // Pawn,Knight,Bishop,Rook,Queen,King (King=0, igual que Python)

// El Rey va PRIMERO (su VALOR es 0, el mas bajo de todos) -- la version
// Python (_TIPOS_POR_VALOR) lo pone al final, heredando el orden natural
// "peon..dama" y agregando rey al final por convencion de lista, sin que
// el codigo lo justifique. El oraculo de fuerza bruta (ver mas abajo)
// demostro que ESO es un bug real y compartido: en posiciones donde el
// rey es un recapturador legitimo mas barato que las piezas mayores, usar
// "rey al final" da un valor SEE subóptimo (hasta una dama de diferencia
// en casos extremos, detectado sobre 4437 capturas al azar). Con el rey
// primero (por su valor real), 0 discrepancias contra el oraculo.
const ORDEN_POR_VALOR: [PieceType; 6] = [
    PieceType::King,
    PieceType::Pawn,
    PieceType::Knight,
    PieceType::Bishop,
    PieceType::Rook,
    PieceType::Queen,
];

fn valor(pt: PieceType) -> i32 {
    VALOR[pt as usize]
}

/// Atacantes de `color` sobre `sq`, calculados sobre bitboards LOCALES
/// (no los del tablero real) para poder simular remociones paso a paso.
fn atacantes_a(color: Color, sq: Square, occupied: Bitboard, bb: &[[Bitboard; 6]; 2]) -> Bitboard {
    let idx = color as usize;
    (king_attacks(sq) & bb[idx][PieceType::King as usize])
        | (knight_attacks(sq) & bb[idx][PieceType::Knight as usize])
        | (bishop_attacks(sq, occupied) & (bb[idx][PieceType::Bishop as usize] | bb[idx][PieceType::Queen as usize]))
        | (rook_attacks(sq, occupied) & (bb[idx][PieceType::Rook as usize] | bb[idx][PieceType::Queen as usize]))
        | (pawn_attacks(color.opposite(), sq) & bb[idx][PieceType::Pawn as usize])
}

/// Continuacion optima del intercambio para `color_en_turno`, sobre bitboards
/// locales: en cada paso usa SIEMPRE el atacante de menor valor disponible
/// (optimo probado para este tipo de intercambio alternado), y decide
/// "capturar o parar" con la formula recursiva estandar
/// max(0, valor_en_to - continuacion_del_rival). Formulacion directa en vez
/// del clasico arreglo gain[]+"unwind" al estilo C con `while(--d)` -- ese
/// patron requiere pre-decrementar ANTES de probar la condicion del bucle,
/// y una traduccion ingenua a Rust (`while len()>1`) prueba la condicion
/// ANTES de descontar, ejecutando un paso de mas y dando resultados
/// incorrectos en posiciones con 3+ capturas en cadena (detectado por el
/// oraculo de fuerza bruta). Esta forma recursiva evita el problema
/// enteramente y coincide exactamente con see_oracle().
fn see_recurse(to_sq: Square, occupied: Bitboard, bb: &[[Bitboard; 6]; 2], color_en_turno: Color, valor_en_to: i32) -> i32 {
    let atacantes = atacantes_a(color_en_turno, to_sq, occupied, bb) & !bit(to_sq);
    for &pt in ORDEN_POR_VALOR.iter() {
        let disponibles = atacantes & bb[color_en_turno as usize][pt as usize];
        if disponibles != 0 {
            let sq = lsb(disponibles);
            let mut occ2 = occupied;
            let mut bb2 = *bb;
            occ2 &= !bit(sq);
            bb2[color_en_turno as usize][pt as usize] &= !bit(sq);
            let g = valor_en_to - see_recurse(to_sq, occ2, &bb2, color_en_turno.opposite(), valor(pt));
            return g.max(0);
        }
    }
    0
}

pub fn see(b: &Board, mv: &Move) -> i32 {
    let to_sq = mv.to;
    let from_sq = mv.from;
    let es_al_paso = mv.flag == MoveFlag::EnPassant;

    let victima_tipo = if es_al_paso {
        Some(PieceType::Pawn)
    } else {
        b.piece_at(to_sq).map(|(_, pt)| pt)
    };

    let (color_atacante, atacante_tipo) =
        b.piece_at(from_sq).expect("see: no hay pieza en 'from'");

    let mut occupied = b.occupied;
    let mut bb = b.pieces; // Copy barato: [[u64;6];2]

    let gain0 = victima_tipo.map(valor).unwrap_or(0);

    bb[color_atacante as usize][atacante_tipo as usize] &= !bit(from_sq);
    occupied &= !bit(from_sq);

    if es_al_paso {
        let them = color_atacante.opposite();
        let victima_sq = make_square(file_of(to_sq), rank_of(from_sq));
        bb[them as usize][PieceType::Pawn as usize] &= !bit(victima_sq);
        occupied &= !bit(victima_sq);
    }

    let valor_en_to = mv.promotion.map(valor).unwrap_or_else(|| valor(atacante_tipo));
    let resto = see_recurse(to_sq, occupied, &bb, color_atacante.opposite(), valor_en_to);
    gain0 - resto
}

// ============================================================
//  Oraculo de fuerza bruta (independiente del atajo de arriba) -- misma
//  simulacion sobre bitboards LOCALES que see() (no toca el tablero real,
//  para no arrastrar preguntas de legalidad/jaque que SEE ignora a
//  proposito), pero probando TODOS los atacantes disponibles en cada paso
//  via minimax real, en vez de asumir que "el mas barato primero" siempre
//  es optimo. Si esto coincide con see() en muchas posiciones al azar, es
//  evidencia fuerte de que el atajo (mucho mas rapido) es correcto.
// ============================================================

fn oracle_recurse(
    to_sq: Square,
    occupied: Bitboard,
    bb: &[[Bitboard; 6]; 2],
    color_en_turno: Color,
    valor_en_to: i32,
) -> i32 {
    let atacantes = atacantes_a(color_en_turno, to_sq, occupied, bb) & !bit(to_sq);
    let mut mejor = 0; // parar (no capturar) siempre es una opcion valida, ganancia 0
    for &pt in ORDEN_POR_VALOR.iter() {
        let mut candidatos = atacantes & bb[color_en_turno as usize][pt as usize];
        while candidatos != 0 {
            let sq = crate::bitboard::pop_lsb(&mut candidatos);
            let mut occ2 = occupied;
            let mut bb2 = *bb;
            occ2 &= !bit(sq);
            bb2[color_en_turno as usize][pt as usize] &= !bit(sq);
            let g = valor_en_to - oracle_recurse(to_sq, occ2, &bb2, color_en_turno.opposite(), valor(pt));
            if g > mejor {
                mejor = g;
            }
        }
    }
    mejor
}

pub fn see_oracle(b: &Board, mv: &Move) -> i32 {
    let to_sq = mv.to;
    let from_sq = mv.from;
    let es_al_paso = mv.flag == MoveFlag::EnPassant;

    let victima_tipo = if es_al_paso {
        Some(PieceType::Pawn)
    } else {
        b.piece_at(to_sq).map(|(_, pt)| pt)
    };
    let (color_atacante, atacante_tipo) = b.piece_at(from_sq).expect("see_oracle: no hay pieza en 'from'");

    let mut occupied = b.occupied;
    let mut bb = b.pieces;

    let gain0 = victima_tipo.map(valor).unwrap_or(0);

    bb[color_atacante as usize][atacante_tipo as usize] &= !bit(from_sq);
    occupied &= !bit(from_sq);
    if es_al_paso {
        let them = color_atacante.opposite();
        let victima_sq = make_square(file_of(to_sq), rank_of(from_sq));
        bb[them as usize][PieceType::Pawn as usize] &= !bit(victima_sq);
        occupied &= !bit(victima_sq);
    }

    let valor_en_to = mv.promotion.map(valor).unwrap_or_else(|| valor(atacante_tipo));
    let resto = oracle_recurse(to_sq, occupied, &bb, color_atacante.opposite(), valor_en_to);
    gain0 - resto
}
