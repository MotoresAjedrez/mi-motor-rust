// NNUE incremental para la evaluacion hibrida.
//
// La primera capa usa features binarias dispersas. El acumulador guarda
// bias + suma de las columnas activas y, al avanzar una posicion, solo suma o
// resta las features que cambiaron. Los pesos actuales conservan el formato
// previo 770 -> 256 -> 32 -> 1, por lo que siguen siendo compatibles.

use crate::board::Board;
use crate::types::{Color, ALL_PIECE_TYPES};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

pub const N_ENTRADA: usize = 770;
const N_OCULTA1: usize = 256;
const N_OCULTA2: usize = 32;

fn checksum_fnv1a(datos: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in datos {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

pub struct RedNeural {
    // Columna j = contribucion de la feature j a las 256 neuronas de la
    // primera capa. Esta disposicion permite actualizar el acumulador con un
    // bloque contiguo al hacer una jugada.
    w1_col: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    w3: Vec<f32>,
    b3: f32,
}

impl RedNeural {
    fn cargar_de_bytes(datos: &[u8]) -> Option<RedNeural> {
        let esperado = (N_OCULTA1 * N_ENTRADA + N_OCULTA1 + N_OCULTA2 * N_OCULTA1 + N_OCULTA2 + N_OCULTA2 + 1) * 4;
        if datos.len() != esperado {
            eprintln!(
                "info string NNUE: tamano de archivo inesperado ({} bytes, se esperaban {})",
                datos.len(), esperado
            );
            return None;
        }
        let mut cursor = 0usize;
        let leer_f32_vec = |n: usize, cursor: &mut usize| -> Vec<f32> {
            let mut valores = Vec::with_capacity(n);
            for _ in 0..n {
                let bytes: [u8; 4] = datos[*cursor..*cursor + 4].try_into().unwrap();
                valores.push(f32::from_le_bytes(bytes));
                *cursor += 4;
            }
            valores
        };
        let w1_fila = leer_f32_vec(N_OCULTA1 * N_ENTRADA, &mut cursor);
        let b1 = leer_f32_vec(N_OCULTA1, &mut cursor);
        let w2 = leer_f32_vec(N_OCULTA2 * N_OCULTA1, &mut cursor);
        let b2 = leer_f32_vec(N_OCULTA2, &mut cursor);
        let w3 = leer_f32_vec(N_OCULTA2, &mut cursor);
        let b3 = leer_f32_vec(1, &mut cursor)[0];

        let todos_finitos = w1_fila
            .iter()
            .chain(b1.iter())
            .chain(w2.iter())
            .chain(b2.iter())
            .chain(w3.iter())
            .chain(std::iter::once(&b3))
            .all(|v| v.is_finite() && v.abs() <= 1.0e6);
        if !todos_finitos {
            eprintln!("info string NNUE: pesos invalidos (NaN, infinito o magnitud absurda)");
            return None;
        }

        let mut w1_col = vec![0.0; N_ENTRADA * N_OCULTA1];
        for fila in 0..N_OCULTA1 {
            for columna in 0..N_ENTRADA {
                w1_col[columna * N_OCULTA1 + fila] = w1_fila[fila * N_ENTRADA + columna];
            }
        }

        Some(RedNeural { w1_col, b1, w2, b2, w3, b3 })
    }

    fn sumar_feature(&self, acumulador: &mut [f32; N_OCULTA1], feature: usize, signo: f32) {
        let columna = &self.w1_col[feature * N_OCULTA1..(feature + 1) * N_OCULTA1];
        for (valor, peso) in acumulador.iter_mut().zip(columna) {
            *valor += signo * peso;
        }
    }

    fn salida(&self, acumulador: &[f32; N_OCULTA1]) -> f32 {
        // ReLU UNA vez por neurona (antes se recalculaba max(0.0) dentro
        // del producto de cada una de las 32 filas de w2: 8192 max en vez
        // de 256).
        let mut h1 = [0.0f32; N_OCULTA1];
        for (v, &a) in h1.iter_mut().zip(acumulador.iter()) {
            *v = a.max(0.0);
        }
        let mut h2 = [0.0; N_OCULTA2];
        for (i, fila) in self.w2.chunks_exact(N_OCULTA1).enumerate() {
            // Producto punto en 8 carriles independientes. La suma f32
            // secuencial (.zip().map().sum()) encadena cada suma con la
            // anterior y el compilador no puede vectorizarla (f32 no es
            // asociativo); con 8 acumuladores separados la dependencia se
            // rompe y LLVM emite NEON. Perfilado con `sample`: esta funcion
            // era ~80% del tiempo de busqueda con NNUE activada.
            let mut lanes = [0.0f32; 8];
            for (wc, vc) in fila.chunks_exact(8).zip(h1.chunks_exact(8)) {
                for j in 0..8 {
                    lanes[j] += wc[j] * vc[j];
                }
            }
            let dot: f32 = lanes.iter().sum();
            h2[i] = (self.b2[i] + dot).max(0.0);
        }
        let dot: f32 = self.w3.iter().zip(h2.iter()).map(|(&peso, &valor)| peso * valor).sum();
        (self.b3 + dot) * 100.0
    }
}

#[derive(Clone)]
pub struct NnueAccumulator {
    red: Arc<RedNeural>,
    primera_capa: [f32; N_OCULTA1],
}

impl NnueAccumulator {
    fn desde_tablero(red: Arc<RedNeural>, b: &Board) -> NnueAccumulator {
        let mut primera_capa = [0.0; N_OCULTA1];
        primera_capa.copy_from_slice(&red.b1);
        for (color_idx, color) in [(0usize, Color::White), (1usize, Color::Black)] {
            for (pt_idx, &pt) in ALL_PIECE_TYPES.iter().enumerate() {
                let mut piezas = b.pieces[color as usize][pt as usize];
                while piezas != 0 {
                    let sq = crate::bitboard::pop_lsb(&mut piezas);
                    red.sumar_feature(&mut primera_capa, feature_pieza(color_idx, pt_idx, sq as usize), 1.0);
                }
            }
        }
        if b.turn == Color::White {
            red.sumar_feature(&mut primera_capa, 768, 1.0);
        }
        if enroque_del_bando(b) {
            red.sumar_feature(&mut primera_capa, 769, 1.0);
        }
        NnueAccumulator { red, primera_capa }
    }

    /// Construye el acumulador hijo actualizando solamente las features que
    /// cambiaron. Incluye capturas, promociones, en passant, enroque, turno
    /// y derechos de enroque sin depender de banderas especiales de jugada.
    pub fn despues_de_jugada(&self, antes: &Board, despues: &Board) -> NnueAccumulator {
        let mut siguiente = self.clone();
        for color in 0..2 {
            for pieza in 0..6 {
                let antes_bb = antes.pieces[color][pieza];
                let despues_bb = despues.pieces[color][pieza];
                let mut salen = antes_bb & !despues_bb;
                while salen != 0 {
                    let sq = crate::bitboard::pop_lsb(&mut salen);
                    siguiente.red.sumar_feature(&mut siguiente.primera_capa, feature_pieza(color, pieza, sq as usize), -1.0);
                }
                let mut entran = despues_bb & !antes_bb;
                while entran != 0 {
                    let sq = crate::bitboard::pop_lsb(&mut entran);
                    siguiente.red.sumar_feature(&mut siguiente.primera_capa, feature_pieza(color, pieza, sq as usize), 1.0);
                }
            }
        }
        actualizar_booleano(&siguiente.red, &mut siguiente.primera_capa, 768, antes.turn == Color::White, despues.turn == Color::White);
        actualizar_booleano(&siguiente.red, &mut siguiente.primera_capa, 769, enroque_del_bando(antes), enroque_del_bando(despues));
        siguiente
    }

    pub fn evaluar(&self) -> f32 {
        self.red.salida(&self.primera_capa)
    }
}

#[inline]
fn feature_pieza(color: usize, pieza: usize, sq: usize) -> usize {
    (color * 6 + pieza) * 64 + sq
}

#[inline]
fn enroque_del_bando(b: &Board) -> bool {
    let derechos = match b.turn {
        Color::White => crate::board::CASTLE_WK | crate::board::CASTLE_WQ,
        Color::Black => crate::board::CASTLE_BK | crate::board::CASTLE_BQ,
    };
    b.castling_rights & derechos != 0
}

fn actualizar_booleano(red: &RedNeural, acumulador: &mut [f32; N_OCULTA1], feature: usize, antes: bool, despues: bool) {
    match (antes, despues) {
        (false, true) => red.sumar_feature(acumulador, feature, 1.0),
        (true, false) => red.sumar_feature(acumulador, feature, -1.0),
        _ => {}
    }
}

static RED: OnceLock<RwLock<Option<Arc<RedNeural>>>> = OnceLock::new();
static ACTIVA: AtomicBool = AtomicBool::new(false);

fn almacenamiento() -> &'static RwLock<Option<Arc<RedNeural>>> {
    RED.get_or_init(|| RwLock::new(None))
}

/// Carga o reemplaza los pesos. Si la ruta o el contenido son invalidos se
/// conserva la red anterior, evitando que un error de escritura apague una
/// NNUE que ya estaba funcionando. Devuelve el checksum FNV-1a del archivo.
///
/// UCI detiene cualquier busqueda antes de ejecutar setoption, asi que
/// reemplazar la Arc no invalida acumuladores que sigan vivos: conservan una
/// referencia a su propia red.
pub fn cargar_detallado(path: &str) -> Result<u64, String> {
    let datos = std::fs::read(path).map_err(|e| format!("no se pudo leer: {e}"))?;
    let checksum = checksum_fnv1a(&datos);
    let red = RedNeural::cargar_de_bytes(&datos)
        .ok_or_else(|| "formato o valores de pesos invalidos".to_string())?;
    *almacenamiento().write().expect("candado NNUE envenenado") = Some(Arc::new(red));
    Ok(checksum)
}

pub fn cargar(path: &str) -> bool {
    cargar_detallado(path).is_ok()
}

pub fn set_activa(valor: bool) {
    ACTIVA.store(valor, Ordering::Relaxed);
}

pub fn esta_activa() -> bool {
    ACTIVA.load(Ordering::Relaxed)
}

pub fn hay_red_cargada() -> bool {
    almacenamiento().read().expect("candado NNUE envenenado").is_some()
}

pub fn crear_acumulador(b: &Board) -> Option<NnueAccumulator> {
    if !ACTIVA.load(Ordering::Relaxed) {
        return None;
    }
    let red = almacenamiento().read().expect("candado NNUE envenenado").clone()?;
    Some(NnueAccumulator::desde_tablero(red, b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Move, MoveFlag};

    #[test]
    fn acumulador_incremental_coincide_con_recalculo() {
        let datos = std::fs::read("nn_weights/pesos_v1.bin").expect("pesos de prueba");
        let red = Arc::new(RedNeural::cargar_de_bytes(&datos).expect("pesos validos"));
        let inicial = Board::startpos();
        let tras_e4 = inicial.make_move(&Move::new(12, 28, None, MoveFlag::DoublePush));

        let incremental = NnueAccumulator::desde_tablero(Arc::clone(&red), &inicial)
            .despues_de_jugada(&inicial, &tras_e4);
        let recalculado = NnueAccumulator::desde_tablero(red, &tras_e4);

        assert!((incremental.evaluar() - recalculado.evaluar()).abs() < 0.01);
    }

    #[test]
    fn rechaza_nan_sin_panico() {
        let mut datos = std::fs::read("nn_weights/pesos_v1.bin").expect("pesos de prueba");
        datos[0..4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(RedNeural::cargar_de_bytes(&datos).is_none());
    }

    #[test]
    fn checksum_es_estable() {
        let datos = std::fs::read("nn_weights/pesos_v1.bin").expect("pesos de prueba");
        assert_eq!(checksum_fnv1a(&datos), checksum_fnv1a(&datos));
        assert_ne!(checksum_fnv1a(&datos), 0);
    }
}
