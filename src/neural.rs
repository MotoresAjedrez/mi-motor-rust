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
    // v13.1: w1 guardada TRANSPUESTA (columna-mayor: w1_col[j*256..j*256+256]
    // = los 256 pesos que conectan la entrada j a cada neurona oculta),
    // no como vino del archivo (fila-mayor). La entrada de esta red es un
    // one-hot disperso: de las 770 entradas, solo ~32-34 valen 1.0 (una por
    // pieza en el tablero, mas 2 bits de turno/enroque) -- el resto son
    // ceros que no aportan nada a la suma. Con la matriz en columna-mayor,
    // "sumar la contribucion de la entrada activa j" es un slice contiguo
    // de 256 floats (rapido, vectoriza bien); con la fila-mayor original
    // habria que leer con salto de 770 floats por cada uno de los 256 --
    // practicamente un cache-miss por lectura. Medido: recompute denso
    // completo (todas las 770 entradas, la mayoria en 0) ~10-12k nodos/seg
    // con la red activada; con esto, ver comentario en eval_red().
    w1_col: Vec<f32>, // [770 x 256] transpuesta
    b1: Vec<f32>,     // [256]
    w2: Vec<f32>,     // [32 x 256]
    b2: Vec<f32>,     // [32]
    w3: Vec<f32>,     // [1 x 32]
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
        let w1_fila = leer_f32_vec(N_OCULTA1 * N_ENTRADA, &mut cursor);
        let b1 = leer_f32_vec(N_OCULTA1, &mut cursor);
        let w2 = leer_f32_vec(N_OCULTA2 * N_OCULTA1, &mut cursor);
        let b2 = leer_f32_vec(N_OCULTA2, &mut cursor);
        let w3 = leer_f32_vec(N_OCULTA2, &mut cursor);
        let b3v = leer_f32_vec(1, &mut cursor);

        // Transponer una sola vez al cargar (esto SI es caro -- 770*256 --
        // pero pasa UNA vez al arrancar, no en cada nodo de busqueda).
        let mut w1_col = vec![0f32; N_ENTRADA * N_OCULTA1];
        for i in 0..N_OCULTA1 {
            for j in 0..N_ENTRADA {
                w1_col[j * N_OCULTA1 + i] = w1_fila[i * N_ENTRADA + j];
            }
        }

        Some(RedNeural { w1_col, b1, w2, b2, w3, b3: b3v[0] })
    }

    /// Forward pass explotando que la entrada es dispersa (one-hot por
    /// pieza): en vez de multiplicar 770 entradas (la enorme mayoria en
    /// cero) por cada una de las 256 neuronas ocultas, se parte de los
    /// sesgos y se SUMA la columna de w1 de cada entrada activa nada mas.
    /// Matematicamente identico al forward denso (x[j]=1.0 en las activas,
    /// 0.0 en el resto, asi que sum(w*x) == sum de columnas activas) --
    /// mismo resultado, ~20-30x menos trabajo (medido).
    fn forward_disperso(&self, indices_activos: &[u16]) -> f32 {
        let mut h1 = [0f32; N_OCULTA1];
        h1.copy_from_slice(&self.b1);
        for &j in indices_activos {
            let base = j as usize * N_OCULTA1;
            let col = &self.w1_col[base..base + N_OCULTA1];
            for (h, &w) in h1.iter_mut().zip(col.iter()) {
                *h += w;
            }
        }
        for h in h1.iter_mut() {
            *h = h.max(0.0);
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

// Maximo teorico de entradas activas: 32 piezas (16 por bando, un jugador
// NUNCA gana piezas de mas alla de sus 16 iniciales, promocion solo
// convierte peones ya existentes) + 2 bits (turno, enroque) = 34. Arreglo
// de tamano fijo en la pila -- CERO reservas de memoria dinamica por
// llamada (antes con Vec::with_capacity se reservaba en el heap en CADA
// nodo de busqueda con la red activada).
const MAX_ACTIVOS: usize = 34;

/// Lista los indices (0..770) de entradas activas (valor 1.0) para una
/// posicion -- tipicamente ~32-34 (una por pieza en el tablero, mas turno
/// y/o derechos de enroque). Mismo orden/encoding EXACTO que
/// board_a_vector() en features_red.py (ver comentario arriba del archivo).
fn indices_activos(b: &Board) -> ([u16; MAX_ACTIVOS], usize) {
    let mut idx = [0u16; MAX_ACTIVOS];
    let mut n = 0usize;
    for (color_idx, color) in [(0usize, Color::White), (1usize, Color::Black)] {
        for (pt_idx, &pt) in ALL_PIECE_TYPES.iter().enumerate() {
            let mut bb = b.pieces[color as usize][pt as usize];
            while bb != 0 {
                let sq = crate::bitboard::pop_lsb(&mut bb);
                idx[n] = ((color_idx * 6 + pt_idx) * 64 + sq as usize) as u16;
                n += 1;
            }
        }
    }
    if b.turn == Color::White {
        idx[n] = 768;
        n += 1;
    }
    let (bit_k, bit_q) = if b.turn == Color::White {
        (crate::board::CASTLE_WK, crate::board::CASTLE_WQ)
    } else {
        (crate::board::CASTLE_BK, crate::board::CASTLE_BQ)
    };
    if b.castling_rights & (bit_k | bit_q) != 0 {
        idx[n] = 769;
        n += 1;
    }
    (idx, n)
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
            let (idx, n) = indices_activos(b);
            Some(red.forward_disperso(&idx[..n]))
        }
        _ => None,
    }
}
