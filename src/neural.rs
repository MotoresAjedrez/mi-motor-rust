// Evaluacion neuronal ligera (v13), hibrida con la evaluacion clasica de
// eval.rs -- NO reemplaza nada, se suma como correccion opcional.
//
// HONESTIDAD TECNICA (misma nota que ~/mi-motor/evaluar_red.py, de donde
// viene la arquitectura): esto NO es NNUE real. NNUE de verdad mantiene un
// "accumulator" que se actualiza de forma INCREMENTAL con cada jugada
// (sumando/restando solo el efecto de la pieza que se movio), sin nunca
// recalcular la red entera desde cero. Ese diseño es justamente lo que hace
// rapidos a los motores NNUE reales (Stockfish, etc.). Ac  no hay
// accumulator incremental: la red se recalcula por completo en cada
// llamada, como en la version Python original. Se la llama "evaluacion
// neuronal ligera" en toda la documentacion para no exagerar lo que es.
//
// Arquitectura (misma que ~/mi-motor/entrenar_red.py, pesos portados
// directamente desde pesos.npz, ver nn_weights/README.md):
//   770 entradas -> 256 (ReLU) -> 32 (ReLU) -> 1 (lineal)
// Entrada: 770 floats = 12 planos de 64 casilleros (6 tipos de pieza x
// blancas, mismos 6 x negras, MISMO orden Peon/Caballo/Alfil/Torre/
// Dama/Rey que PieceType) + 1 bit de turno + 1 bit de derechos de enroque
// del bando que mueve. Mismo encoding EXACTO que features_red.py en Python
// (verificado: make_square(file,rank)=rank*8+file coincide con la
// numeracion de python-chess a1=0..h8=63) -- si este encoding se
// desalinea del que uso el entrenamiento, la red da numeros sin sentido
// aunque compile y corra sin errores.

use crate::board::Board;
use crate::types::{Color, ALL_PIECE_TYPES};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

pub const N_ENTRADA: usize = 770;
const N_OCULTA1: usize = 256;
const N_OCULTA2: usize = 32;

pub struct RedNeural {
    w1: Vec<f32>, // [256 x 770], fila i = pesos de la neurona oculta i
    b1: Vec<f32>, // [256]
    w2: Vec<f32>, // [32 x 256]
    b2: Vec<f32>, // [32]
    w3: Vec<f32>, // [1 x 32]
    b3: f32,
}

impl RedNeural {
    fn cargar_de_bytes(datos: &[u8]) -> Option<RedNeural> {
        let esperado = (N_OCULTA1 * N_ENTRADA + N_OCULTA1 + N_OCULTA2 * N_OCULTA1 + N_OCULTA2 + N_OCULTA2 + 1) * 4;
        if datos.len() != esperado {
            eprintln!(
                "info string red neuronal: tamano de archivo inesperado ({} bytes, se esperaban {}) -- no se activa",
                datos.len(),
                esperado
            );
            return None;
        }
        let mut cursor = 0usize;
        let leer_f32_vec = |n: usize, cursor: &mut usize| -> Vec<f32> {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                let b: [u8; 4] = datos[*cursor..*cursor + 4].try_into().unwrap();
                v.push(f32::from_le_bytes(b));
                *cursor += 4;
            }
            v
        };
        let w1 = leer_f32_vec(N_OCULTA1 * N_ENTRADA, &mut cursor);
        let b1 = leer_f32_vec(N_OCULTA1, &mut cursor);
        let w2 = leer_f32_vec(N_OCULTA2 * N_OCULTA1, &mut cursor);
        let b2 = leer_f32_vec(N_OCULTA2, &mut cursor);
        let w3 = leer_f32_vec(N_OCULTA2, &mut cursor);
        let b3v = leer_f32_vec(1, &mut cursor);
        Some(RedNeural { w1, b1, w2, b2, w3, b3: b3v[0] })
    }

    /// Forward pass manual (sin dependencias externas) -- devuelve
    /// centipeones desde la perspectiva del bando que mueve, misma
    /// convencion que evaluate() en eval.rs (y que evaluar_red.py: las
    /// etiquetas de entrenamiento se generaron con score.pov(board.turn)).
    fn forward(&self, x: &[f32; N_ENTRADA]) -> f32 {
        // Dot products con iteradores (zip), no indexado manual: deja que
        // LLVM elimine los chequeos de rango y vectorice el bucle -- el
        // indexado manual (fila[j]) midio ~150x mas lento en la practica
        // (no vectorizaba pese a opt-level=3/lto=fat en Cargo.toml).
        let mut h1 = [0f32; N_OCULTA1];
        for (i, fila) in self.w1.chunks_exact(N_ENTRADA).enumerate() {
            let dot: f32 = fila.iter().zip(x.iter()).map(|(&w, &xi)| w * xi).sum();
            h1[i] = (self.b1[i] + dot).max(0.0);
        }
        let mut h2 = [0f32; N_OCULTA2];
        for (i, fila) in self.w2.chunks_exact(N_OCULTA1).enumerate() {
            let dot: f32 = fila.iter().zip(h1.iter()).map(|(&w, &hi)| w * hi).sum();
            h2[i] = (self.b2[i] + dot).max(0.0);
        }
        let dot: f32 = self.w3.iter().zip(h2.iter()).map(|(&w, &hi)| w * hi).sum();
        (self.b3 + dot) * 100.0
    }
}

/// Construye el vector de entrada de 770 floats para una posicion. Mismo
/// orden EXACTO que board_a_vector() en features_red.py.
pub fn vector_entrada(b: &Board) -> [f32; N_ENTRADA] {
    let mut v = [0f32; N_ENTRADA];
    for (color_idx, color) in [(0usize, Color::White), (1usize, Color::Black)] {
        for (pt_idx, &pt) in ALL_PIECE_TYPES.iter().enumerate() {
            let mut bb = b.pieces[color as usize][pt as usize];
            while bb != 0 {
                let sq = crate::bitboard::pop_lsb(&mut bb);
                v[(color_idx * 6 + pt_idx) * 64 + sq as usize] = 1.0;
            }
        }
    }
    v[768] = if b.turn == Color::White { 1.0 } else { 0.0 };
    let (bit_k, bit_q) = if b.turn == Color::White {
        (crate::board::CASTLE_WK, crate::board::CASTLE_WQ)
    } else {
        (crate::board::CASTLE_BK, crate::board::CASTLE_BQ)
    };
    v[769] = if b.castling_rights & (bit_k | bit_q) != 0 { 1.0 } else { 0.0 };
    v
}

static RED: OnceLock<Option<RedNeural>> = OnceLock::new();
static ACTIVA: AtomicBool = AtomicBool::new(false);

/// Intenta cargar los pesos desde `path` (llamado una sola vez, desde
/// "setoption name NNPath" o al arrancar si hay ruta por defecto). Si el
/// archivo no existe o esta corrupto, la red simplemente queda sin cargar
/// (None) y eval_red() siempre devuelve None -- nunca hace panic ni
/// bloquea el arranque del motor por esto.
pub fn cargar(path: &str) -> bool {
    let red = std::fs::read(path).ok().and_then(|datos| RedNeural::cargar_de_bytes(&datos));
    let ok = red.is_some();
    let _ = RED.set(red);
    ok
}

pub fn set_activa(v: bool) {
    ACTIVA.store(v, Ordering::Relaxed);
}

pub fn activa() -> bool {
    ACTIVA.load(Ordering::Relaxed)
}

/// Evalua con la red si esta activada (UCI "UseNN") Y los pesos se
/// cargaron con exito. Devuelve None en cualquier otro caso -- el llamador
/// (evaluate() en eval.rs) debe tratar None como "no sumar nada", igual
/// que como se maneja SyzygyPath/BookPath cuando no estan configurados.
pub fn eval_red(b: &Board) -> Option<f32> {
    if !activa() {
        return None;
    }
    match RED.get() {
        Some(Some(red)) => {
            let x = vector_entrada(b);
            Some(red.forward(&x))
        }
        _ => None,
    }
}
