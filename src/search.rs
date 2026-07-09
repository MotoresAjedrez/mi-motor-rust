// Negamax + poda alfa-beta + iterative deepening + quiescence + TT.
// Primera version jugable de la Fase 3: SEE, null-move, killers/history y
// LMR quedan para una siguiente pasada si el tiempo alcanza (documentado
// en el reporte final de la sesion).

use crate::board::Board;
use crate::eval::evaluate;
use crate::movegen::{generate_legal, generate_pseudo_legal};
use crate::types::{Move, MoveFlag};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub const INFINITO: i32 = 30_000;
pub const MATE: i32 = 29_000;
const MAX_PLY: u32 = 64;

// Contempt dinamico: en vez de puntuar toda tabla (repeticion/regla de 50)
// como exactamente 0, se mide la evaluacion estatica de la posicion desde la
// perspectiva de quien mueve. Si esta claramente ganando (por encima del
// umbral), la tabla se puntua PEOR que 0 -- para que la busqueda la evite
// activamente cuando hay alternativas de progreso, en vez de solo reconocerla
// una vez que ya esta ahi. Si esta claramente perdiendo, se puntua MEJOR que
// 0 -- para que la busque activamente como recurso defensivo. Es una funcion
// de la posicion (no depende del historial de jugadas), asi que es seguro
// guardarla en la TT igual que cualquier otro puntaje.
const CONTEMPT_UMBRAL: i32 = 500;
const CONTEMPT_PENALIZACION: i32 = 200;

fn draw_score(b: &Board) -> i32 {
    let se = evaluate(b);
    if se > CONTEMPT_UMBRAL {
        -CONTEMPT_PENALIZACION
    } else if se < -CONTEMPT_UMBRAL {
        CONTEMPT_PENALIZACION
    } else {
        0
    }
}

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
pub struct TTEntry {
    key: u64,
    depth: i32,
    score: i32,
    flag: TTFlag,
    best: Option<Move>,
}

pub struct TimeUp;

const MAX_KILLER_PLY: usize = 100; // margen sobre MAX_PLY para cubrir extensiones de jaque

// 6 tipos de pieza x 64 casilleros, dos veces (jugada rival + jugada propia).
const CONT_HIST_SIZE: usize = 6 * 64 * 6 * 64;

#[inline]
fn cont_idx(prev_pt: usize, prev_to: usize, pt: usize, to: usize) -> usize {
    ((prev_pt * 64 + prev_to) * 6 + pt) * 64 + to
}

// TT compartida entre hilos (Lazy SMP): un Mutex POR CASILLERO, no uno solo
// para toda la tabla -- bloquear la tabla entera en cada sondeo/guardado
// (que pasa en CADA nodo) seria un cuello de botella tan grande que
// anularia la ganancia de buscar en paralelo. Con un mutex por casillero,
// dos hilos solo compiten de verdad si sus busquedas chocan en el MISMO
// indice de la tabla al mismo tiempo, algo relativamente raro con una
// tabla de tamano razonable.
pub type SharedTT = Vec<Mutex<Option<TTEntry>>>;

pub fn construir_tt(tt_mb: usize) -> (Arc<SharedTT>, usize) {
    let entry_size = std::mem::size_of::<Option<TTEntry>>().max(1);
    let mut n_entries = (tt_mb * 1024 * 1024 / entry_size).max(1024);
    n_entries = n_entries.next_power_of_two() >> 1; // asegurar potencia de 2 sin pasarse
    let tt: SharedTT = (0..n_entries).map(|_| Mutex::new(None)).collect();
    (Arc::new(tt), n_entries - 1)
}

pub struct Searcher {
    tt: Arc<SharedTT>,
    tt_mask: usize,
    pub nodes: u64,
    deadline: Option<Instant>,
    stop: bool,
    // killers son validos solo dentro de esta busqueda (por ply del arbol
    // actual); history SI persiste entre jugadas de la partida, igual que la TT.
    killers: Vec<[Option<Move>; 2]>,
    history: Box<[[i32; 64]; 64]>, // [from][to] -- arreglo plano, mas rapido que un HashMap aqui
    // Continuation history ("counter-move history"): a diferencia de history
    // (que solo sabe "esta jugada [from][to] corto mucho, en general"), esta
    // tabla sabe "esta jugada [pieza][to] corto mucho DESPUES de que el
    // rival jugara [pieza][to]" -- captura respuestas tacticas especificas a
    // una jugada rival concreta (p.ej. recapturas, bloqueos de jaque) que el
    // history plano no distingue del resto. Indexada
    // [pieza_rival][casillero_rival][pieza_propia][casillero_propio],
    // aplanada en un Vec para evitar arrays anidados de tamano fijo.
    cont_history: Vec<i32>,
    pub modo_lmr: bool,
    // Desactivable solo para comparacion A/B en pruebas (MIMOTOR_NO_ASPIRATION=1)
    // -- en juego real siempre queda activado, la tecnica en si es segura por
    // construccion (ensancha hasta ventana completa si hace falta).
    pub modo_aspiration: bool,
    // Singular extensions: APAGADO por defecto (al reves que LMR/aspiration).
    // Solo se activa con MIMOTOR_SINGULAR=1 -- exige el test dedicado
    // (comando CLI "singulartest") verde antes de proponer cambiar el
    // default, por pedido explicito: es de las tecnicas mas propensas a
    // bugs sutiles y no se activa "por si acaso".
    pub modo_singular: bool,
    // Historial de repeticion: claves Zobrist de la PARTIDA REAL (persiste
    // entre llamadas a go, la maneja el loop UCI) + las de la linea actual
    // de busqueda (crece/decrece durante la recursion, como el "self.hist"
    // de Python). No se usa la TT para esto porque una entrada de TT no
    // sabe CUANTAS veces se visito esa posicion en esta partida especifica.
    game_history: Vec<u64>,
    path: Vec<u64>,
    pub lmr_intentos: u64,
    pub lmr_reintentos: u64,
    // Lazy SMP: si esta activo, este hilo intercambia las 2 primeras
    // jugadas del orden en la RAIZ (una vez, al armar el orden inicial) para
    // no explorar exactamente la misma linea primero que los demas hilos.
    pub variante_orden_raiz: bool,
    // Bandera compartida con el hilo principal del loop UCI: permite que el
    // comando "stop" interrumpa una busqueda en curso (que corre en su
    // propio hilo -- ver uci_loop en main.rs) sin depender solo del deadline
    // de tiempo. Necesario para "go infinite" y para cumplir el protocolo
    // UCI que exigen los testers de listas de rating como CCRL.
    external_stop: Option<Arc<AtomicBool>>,
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
        let (tt, tt_mask) = construir_tt(tt_mb);
        Searcher {
            tt,
            tt_mask,
            nodes: 0,
            deadline: None,
            stop: false,
            killers: vec![[None, None]; MAX_KILLER_PLY],
            history: Box::new([[0i32; 64]; 64]),
            cont_history: vec![0i32; CONT_HIST_SIZE],
            // Activado por defecto: el torneo h2h de esta sesion confirmo
            // +80 ELO (61.3% en 40 partidas) con la reescritura PVS -- ver
            // resultados_lmr_h2h.txt en ~/mi-motor. MIMOTOR_LMR=0 lo desactiva
            // explicitamente para pruebas comparativas.
            modo_lmr: std::env::var("MIMOTOR_LMR").as_deref() != Ok("0"),
            modo_aspiration: std::env::var("MIMOTOR_NO_ASPIRATION").as_deref() != Ok("1"),
            modo_singular: std::env::var("MIMOTOR_SINGULAR").as_deref() == Ok("1"),
            game_history: Vec::new(),
            path: Vec::new(),
            lmr_intentos: 0,
            lmr_reintentos: 0,
            variante_orden_raiz: false,
            external_stop: None,
        }
    }

    /// Crea un Searcher que comparte la TT (Arc clonado, mismo mask) de otro
    /// -- para Lazy SMP, donde varios hilos buscan sobre la misma tabla.
    /// Killers/history/game_history quedan LOCALES de este hilo (no tiene
    /// sentido compartirlos, cada hilo ordena sus propias jugadas).
    pub fn new_con_tt_compartida(tt: Arc<SharedTT>, tt_mask: usize, modo_lmr: bool) -> Searcher {
        Searcher {
            tt,
            tt_mask,
            nodes: 0,
            deadline: None,
            stop: false,
            killers: vec![[None, None]; MAX_KILLER_PLY],
            history: Box::new([[0i32; 64]; 64]),
            cont_history: vec![0i32; CONT_HIST_SIZE],
            modo_lmr,
            modo_aspiration: std::env::var("MIMOTOR_NO_ASPIRATION").as_deref() != Ok("1"),
            modo_singular: std::env::var("MIMOTOR_SINGULAR").as_deref() == Ok("1"),
            game_history: Vec::new(),
            path: Vec::new(),
            lmr_intentos: 0,
            lmr_reintentos: 0,
            variante_orden_raiz: false,
            external_stop: None,
        }
    }

    /// Fija (o quita) la bandera compartida de "stop" externo -- se llama
    /// antes de lanzar la busqueda en su propio hilo desde uci_loop.
    pub fn set_external_stop(&mut self, flag: Option<Arc<AtomicBool>>) {
        self.external_stop = flag;
    }

    fn registrar_corte(&mut self, mv: Move, ply: u32, depth: i32, prev: Option<(usize, usize)>, pt_mv: usize) {
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
        if let Some((prev_pt, prev_to)) = prev {
            let idx = cont_idx(prev_pt, prev_to, pt_mv, mv.to as usize);
            self.cont_history[idx] += depth * depth;
        }
    }

    /// Fija el historial de claves Zobrist de la PARTIDA REAL hasta la
    /// posicion actual (lo arma el loop UCI a partir de "position ...
    /// moves ..."). Se llama antes de cada busqueda para que la deteccion
    /// de repeticion vea jugadas ya ocurridas en la partida, no solo las
    /// que aparezcan dentro del arbol de esta busqueda.
    pub fn set_game_history(&mut self, hist: Vec<u64>) {
        self.game_history = hist;
    }

    /// Reconstruye la linea principal (PV) caminando la TT desde `b`,
    /// siguiendo la mejor jugada guardada en cada posicion. Se corta por
    /// `max_len`, por no encontrar entrada en la TT, o por repeticion de
    /// zobrist (posible en ciclos/tablas) -- nunca deberia colgarse.
    /// Uso: solo para mostrar informacion (modo "simple", UCI "info pv"),
    /// no participa de la busqueda en si.
    pub fn extraer_pv(&self, b: &Board, max_len: usize) -> Vec<Move> {
        let mut pv = Vec::with_capacity(max_len);
        let mut vistos = Vec::with_capacity(max_len);
        let mut actual = *b;
        for _ in 0..max_len {
            if vistos.contains(&actual.zobrist) {
                break;
            }
            vistos.push(actual.zobrist);
            let mv = match self.tt_probe(actual.zobrist).and_then(|e| e.best) {
                Some(mv) => mv,
                None => break,
            };
            if !generate_legal(&actual).contains(&mv) {
                break;
            }
            pv.push(mv);
            actual = actual.make_move(&mv);
        }
        pv
    }

    fn tt_index(&self, key: u64) -> usize {
        (key as usize) & self.tt_mask
    }

    fn tt_probe(&self, key: u64) -> Option<TTEntry> {
        let idx = self.tt_index(key);
        match *self.tt[idx].lock().unwrap() {
            Some(e) if e.key == key => Some(e),
            _ => None,
        }
    }

    // v12: politica de reemplazo "prefiere profundidad" -- antes se
    // sobreescribia SIEMPRE sin condicion, asi que una entrada profunda (cara
    // de calcular, muy valiosa) se podia perder por una superficial que cayo
    // en el mismo casillero (quiescence guarda con depth<=0, por ejemplo).
    // Ahora solo se reemplaza si la entrada nueva es igual o mas profunda que
    // la que ya esta (o el casillero esta vacio) -- protege las entradas
    // caras sin necesitar un esquema de "generacion/antiguedad" mas complejo.
    fn tt_store(&mut self, key: u64, depth: i32, score: i32, flag: TTFlag, best: Option<Move>) {
        let idx = self.tt_index(key);
        let mut slot = self.tt[idx].lock().unwrap();
        let reemplazar = match *slot {
            None => true,
            Some(existing) => depth >= existing.depth,
        };
        if reemplazar {
            *slot = Some(TTEntry { key, depth, score, flag, best });
        }
    }

    fn check_time(&mut self) -> Result<(), TimeUp> {
        self.nodes += 1;
        if !self.stop && self.nodes & 1023 == 0 {
            if let Some(dl) = self.deadline {
                if Instant::now() >= dl {
                    self.stop = true;
                }
            }
            if !self.stop {
                if let Some(flag) = &self.external_stop {
                    if flag.load(Ordering::Relaxed) {
                        self.stop = true;
                    }
                }
            }
        }
        if self.stop {
            Err(TimeUp)
        } else {
            Ok(())
        }
    }

    fn order_moves(&self, b: &Board, moves: &mut Vec<Move>, tt_move: Option<Move>) {
        self.order_moves_ply(b, moves, tt_move, MAX_KILLER_PLY as u32, None);
    }

    /// Igual que order_moves pero ademas usa killers/history (por ply) para
    /// ordenar las jugadas silenciosas -- capturas/TT siguen mandando.
    /// `prev` es (pieza, casillero_destino) de la jugada rival que llevo a
    /// esta posicion (None si no se conoce, p.ej. en la raiz) -- alimenta la
    /// continuation history para las jugadas silenciosas.
    fn order_moves_ply(&self, b: &Board, moves: &mut Vec<Move>, tt_move: Option<Move>, ply: u32, prev: Option<(usize, usize)>) {
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
                let h = self.history[mv.from as usize][mv.to as usize];
                let ch = match prev {
                    Some((prev_pt, prev_to)) => {
                        let pt_mv = b.piece_at(mv.from).map(|(_, pt)| pt as usize).unwrap_or(0);
                        self.cont_history[cont_idx(prev_pt, prev_to, pt_mv, mv.to as usize)]
                    }
                    None => 0,
                };
                -(h + ch)
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

        // v12: ademas de capturas, se incluyen promociones a dama SIN captura
        // (peon que corona caminando). Sin esto, un peon a un paso de coronar
        // podia quedar fuera de quiescence por completo (solo capturas) --
        // efecto horizonte clasico en finales de peones pasados: la busqueda
        // principal SI ve la coronacion (genera todas las jugadas legales),
        // pero si quiescence corta ahi antes de esa jugada especifica, evalua
        // con stand_pat sin haber "visto" que el peon corona. Sub-promociones
        // (a torre/alfil/caballo) casi nunca son mejores que coronar dama, no
        // vale la pena el costo de tambien incluirlas aqui.
        let mut moves: Vec<Move> = generate_pseudo_legal(b)
            .into_iter()
            .filter(|m| m.is_capture() || m.promotion == Some(crate::types::PieceType::Queen))
            .collect();
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

    // `en_sondeo_se`: true si este nodo ya es descendiente de una busqueda
    // de VERIFICACION de singular extensions. Critico: sin este freno, cada
    // nodo dentro de esa verificacion podria a su vez lanzar su PROPIA
    // verificacion (y esos, la suya), multiplicando el trabajo en cadena en
    // vez de sumarlo -- confirmado en la practica (una posicion tardo mas de
    // 9 minutos en profundidad fija 9 antes de este freno). Una vez que un
    // nodo entra en modo verificacion, TODOS sus descendientes lo heredan
    // (se propaga, no se resetea a cada paso) y ninguno intenta su propia
    // singular extension.
    fn negamax(&mut self, b: &Board, mut depth: i32, mut alpha: i32, beta: i32, ply: u32, prev: Option<(usize, usize)>, en_sondeo_se: bool) -> Result<i32, TimeUp> {
        self.check_time()?;

        if b.halfmove_clock >= 100 {
            return Ok(draw_score(b));
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
                return Ok(draw_score(b));
            }
        }

        // Tabla de finales: si la posicion ya esta cubierta (pocas piezas),
        // el WDL es un resultado EXACTO -- ganada/tablas/perdida bajo juego
        // perfecto, tratado como "despues de una jugada que reinicia la
        // regla de 50" (probe_wdl_after_zeroing), la forma segura de usarlo
        // dentro del arbol de busqueda. No se prueba en la raiz (eso lo
        // maneja search_time/search_fixed_depth aparte, via DTZ, para elegir
        // la jugada que progresa de verdad, no solo el resultado).
        if ply > 0 {
            if let Some(wdl) = crate::syzygy::probe_wdl(b) {
                return Ok(wdl);
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
        let mut tt_entry_full: Option<TTEntry> = None;
        if let Some(entry) = self.tt_probe(key) {
            tt_move = entry.best;
            tt_entry_full = Some(entry);
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
            let sc_null = -self.negamax(&next, depth - 1 - NULL_MOVE_R, -beta, -beta + 1, ply + 1, None, en_sondeo_se)?;
            if sc_null >= beta {
                return Ok(beta);
            }
        }

        let mut moves = generate_legal(b);
        if moves.is_empty() {
            return Ok(if en_jaque { -MATE + ply as i32 } else { 0 });
        }
        self.order_moves_ply(b, &mut moves, tt_move, ply, prev);
        self.path.push(b.zobrist);

        // Singular extensions: si la jugada de la TT es tan claramente
        // superior a TODAS las demas que ninguna otra logra siquiera
        // acercarse a su puntaje (verificado con una busqueda reducida,
        // ventana nula, sobre el RESTO de las jugadas), esa jugada es
        // "singular" -- la unica opcion real en la posicion -- y merece 1
        // ply extra de profundidad real en vez de recortarse igual que
        // cualquier otra. Apagado por defecto (modo_singular), ver comentario
        // en la definicion del campo.
        // v12: encontrados DOS desvios del algoritmo estandar que explican
        // la explosion medida en v11 (no era la tecnica, era la condicion de
        // activacion demasiado permisiva):
        //  1) Aceptaba entradas TT con flag Exact ademas de Beta. Beta
        //     (fail-high real, cota inferior) es la UNICA que tiene sentido
        //     para esta prueba -- es la que dice "esta jugada ya demostro
        //     ser >= beta". Exact son nodos PV normales (alpha<score<beta),
        //     mucho mas frecuentes que los Beta, y probarlos multiplicaba la
        //     cantidad de nodos que disparaban la sonda por todo el arbol.
        //  2) No excluia jaque: en posiciones con jaque el numero de
        //     respuestas legales suele ser bajo (casi cualquier jugada
        //     "parece" singular) y la extension de jaque ya existente puede
        //     encadenarse con la sonda de verificacion, multiplicando el
        //     costo sin aportar nada (la extension de jaque ya cubre ese caso).
        const SE_PROF_MIN: i32 = 8;
        let mut jugada_singular: Option<Move> = None;
        if self.modo_singular && !en_sondeo_se && !en_jaque && ply > 0 && depth >= SE_PROF_MIN {
            if let (Some(entry), Some(tmv)) = (tt_entry_full, tt_move) {
                if entry.depth >= depth - 3
                    && entry.flag == TTFlag::Beta
                    && entry.score.abs() < MATE - 1000
                    && moves.contains(&tmv)
                {
                    let margen = 2 * depth;
                    let sbeta = entry.score - margen;
                    let sdepth = (depth - 1) / 2;
                    let mut mejor_otra = -INFINITO;
                    let mut se_timed_out = false;
                    for mv in &moves {
                        if *mv == tmv {
                            continue;
                        }
                        let next = b.make_move(mv);
                        match self.negamax(&next, sdepth, -sbeta, -sbeta + 1, ply + 1, None, true) {
                            Ok(v) => {
                                let sc = -v;
                                if sc > mejor_otra {
                                    mejor_otra = sc;
                                }
                                if mejor_otra >= sbeta {
                                    break; // otra jugada ya alcanza la ventana: no es singular
                                }
                            }
                            Err(_) => {
                                se_timed_out = true;
                                break;
                            }
                        }
                    }
                    if !se_timed_out && mejor_otra < sbeta {
                        jugada_singular = Some(tmv);
                    }
                }
            }
        }

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

            let pt_mv = b.piece_at(mv.from).map(|(_, pt)| pt as usize).unwrap_or(0);
            let next = b.make_move(mv);
            let child_prev = Some((pt_mv, mv.to as usize));
            let ext = if jugada_singular == Some(*mv) { 1 } else { 0 };
            let sc = if es_reducible && !next.in_check(next.turn) {
                self.lmr_intentos += 1;
                let r = 1i32.min(depth - 2);
                // PVS real: el sondeo reducido usa ventana NULA (-alpha-1,-alpha)
                // -- solo pregunta "esto es mejor que lo que ya tengo?", no
                // cuanto mejor. Es un bound, no un valor exacto: si supera
                // alfa, no se confia en el numero, se re-busca a profundidad
                // Y ventana completas para obtener el valor real.
                let sondeo = -self.negamax(&next, depth - 1 + ext - r, -alpha - 1, -alpha, ply + 1, child_prev, en_sondeo_se)?;
                if sondeo > alpha {
                    self.lmr_reintentos += 1;
                    -self.negamax(&next, depth - 1 + ext, -beta, -alpha, ply + 1, child_prev, en_sondeo_se)?
                } else {
                    sondeo
                }
            } else {
                -self.negamax(&next, depth - 1 + ext, -beta, -alpha, ply + 1, child_prev, en_sondeo_se)?
            };

            if sc > best_score {
                best_score = sc;
                best_move = Some(*mv);
            }
            if sc > alpha {
                alpha = sc;
            }
            if alpha >= beta {
                self.registrar_corte(*mv, ply, depth, prev, pt_mv);
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
        let mut mejor_sc: i32 = -INFINITO;
        for d in 1..=depth {
            let moves = generate_legal(b);
            if moves.is_empty() {
                break;
            }
            let mut ordered = moves.clone();
            self.order_moves_ply(b, &mut ordered, mejor_mv, 0, None);

            const VENTANA_INICIAL: i32 = 50;
            let (mut vent_alpha, mut vent_beta) =
                if self.modo_aspiration && d >= 2 && mejor_sc.abs() < MATE - 1000 && mejor_sc > -INFINITO {
                    (mejor_sc - VENTANA_INICIAL, mejor_sc + VENTANA_INICIAL)
                } else {
                    (-INFINITO, INFINITO)
                };
            let mut actual_mv = ordered[0];
            let mut actual_sc;
            let mut ancho = VENTANA_INICIAL;
            loop {
                let mut alpha = vent_alpha;
                actual_mv = ordered[0];
                actual_sc = -INFINITO;
                self.path.push(b.zobrist);
                let mut interrumpido = false;
                for mv in &ordered {
                    let pt_mv = b.piece_at(mv.from).map(|(_, pt)| pt as usize).unwrap_or(0);
                    let next = b.make_move(mv);
                    let sc = match self.negamax(&next, d - 1, -vent_beta, -alpha, 1, Some((pt_mv, mv.to as usize)), false) {
                        Ok(v) => -v,
                        Err(_) => {
                            interrumpido = true;
                            break;
                        }
                    };
                    if sc > actual_sc {
                        actual_sc = sc;
                        actual_mv = *mv;
                    }
                    if sc > alpha {
                        alpha = sc;
                    }
                    if alpha >= vent_beta {
                        break;
                    }
                }
                self.path.pop();
                if interrumpido {
                    return (mejor_mv.or(Some(actual_mv)), mejor_sc, self.nodes);
                }
                if actual_sc <= vent_alpha && vent_alpha > -INFINITO {
                    ancho = ancho.saturating_mul(2);
                    vent_alpha = mejor_sc.saturating_sub(ancho).max(-INFINITO);
                    continue;
                }
                if actual_sc >= vent_beta && vent_beta < INFINITO {
                    ancho = ancho.saturating_mul(2);
                    vent_beta = mejor_sc.saturating_add(ancho).min(INFINITO);
                    continue;
                }
                break;
            }
            mejor_mv = Some(actual_mv);
            mejor_sc = actual_sc;
        }
        (mejor_mv, mejor_sc, self.nodes)
    }

    /// Busqueda con presupuesto de tiempo (para UCI "go movetime").
    /// `movetime_ms = None` significa busqueda SIN limite de tiempo propio
    /// (modo "go infinite"): solo termina por `max_depth`, por encontrar un
    /// mate, o porque el hilo UCI activa `external_stop` al recibir "stop".
    pub fn search_time(&mut self, b: &Board, movetime_ms: Option<u64>, max_depth: i32, mut on_info: impl FnMut(i32, i32, u64, u64)) -> (Option<Move>, i32, i32) {
        self.nodes = 0;
        self.stop = false;
        self.killers = vec![[None, None]; MAX_KILLER_PLY];
        self.path = self.game_history.clone();

        // Libro de aperturas: se consulta para CUALQUIER turno (blancas o
        // negras -- la clave Polyglot ya codifica de quien es el turno), no
        // solo cuando el motor abre la partida.
        if let Some(mv) = crate::polyglot::probe(b) {
            on_info(1, 0, 0, 0);
            return (Some(mv), 0, 1);
        }

        // Tabla de finales en la raiz: DTZ da la jugada que progresa de
        // verdad hacia el resultado optimo (no solo "no perder"), asi que
        // reemplaza directamente lo que hubiera elegido la busqueda normal
        // -- sin esto, alfa-beta con WDL exacto en las hojas puede quedar
        // indiferente entre varias jugadas que dan el mismo resultado
        // (todas "ganadas"), incluida una que no progresa nunca.
        if let Some((mv, sc)) = crate::syzygy::mejor_jugada_raiz(b) {
            on_info(1, sc, 0, 0);
            return (Some(mv), sc, 1);
        }

        let inicio = Instant::now();
        self.deadline = movetime_ms.map(|ms| {
            let budget = ms.saturating_sub(30).max(10);
            inicio + std::time::Duration::from_millis(budget)
        });

        let mut mejor_mv: Option<Move> = None;
        let mut mejor_sc: i32 = 0;
        let mut mejor_prof = 0;

        for d in 1..=max_depth {
            let moves = generate_legal(b);
            if moves.is_empty() {
                break;
            }
            let mut ordered = moves.clone();
            self.order_moves_ply(b, &mut ordered, mejor_mv, 0, None);
            if self.variante_orden_raiz && ordered.len() >= 2 {
                ordered.swap(0, 1);
            }

            // Aspiration windows: a partir de la 2da profundidad ya hay un
            // puntaje de referencia (el de la iteracion anterior), asi que en
            // vez de arrancar con ventana completa (-inf,+inf) se arranca
            // angosta alrededor de ese valor -- casi siempre alcanza y poda
            // mucho mas en las subramas, y si falla (la posicion cambio mas
            // de lo esperado) se ensancha y se repite. Nunca cambia la
            // jugada final elegida, solo cuanto cuesta encontrarla.
            const VENTANA_INICIAL: i32 = 50;
            let (mut vent_alpha, mut vent_beta) = if self.modo_aspiration && d >= 2 && mejor_sc.abs() < MATE - 1000 {
                (mejor_sc - VENTANA_INICIAL, mejor_sc + VENTANA_INICIAL)
            } else {
                (-INFINITO, INFINITO)
            };

            let mut actual_mv = ordered[0];
            let mut actual_sc = -INFINITO;
            let mut timed_out = false;
            let mut ancho = VENTANA_INICIAL;

            loop {
                let mut alpha = vent_alpha;
                actual_mv = ordered[0];
                actual_sc = -INFINITO;
                self.path.push(b.zobrist);
                for mv in &ordered {
                    let pt_mv = b.piece_at(mv.from).map(|(_, pt)| pt as usize).unwrap_or(0);
                    let next = b.make_move(mv);
                    match self.negamax(&next, d - 1, -vent_beta, -alpha, 1, Some((pt_mv, mv.to as usize)), false) {
                        Ok(v) => {
                            let sc = -v;
                            if sc > actual_sc {
                                actual_sc = sc;
                                actual_mv = *mv;
                            }
                            if sc > alpha {
                                alpha = sc;
                            }
                            if alpha >= vent_beta {
                                break; // fail-high contra la ventana: cortar y reintentar mas ancho
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
                // Ensanchado exponencial (duplica cada reintento) con techo
                // en ventana completa -- garantiza terminar y converge rapido
                // incluso si la primera estimacion estaba muy lejos.
                if actual_sc <= vent_alpha && vent_alpha > -INFINITO {
                    ancho = ancho.saturating_mul(2);
                    vent_alpha = mejor_sc.saturating_sub(ancho).max(-INFINITO);
                    continue;
                }
                if actual_sc >= vent_beta && vent_beta < INFINITO {
                    ancho = ancho.saturating_mul(2);
                    vent_beta = mejor_sc.saturating_add(ancho).min(INFINITO);
                    continue;
                }
                break; // adentro de la ventana (o ya en ventana completa): valor confiable
            }
            if timed_out {
                break;
            }
            mejor_mv = Some(actual_mv);
            mejor_sc = actual_sc;
            mejor_prof = d;
            on_info(d, mejor_sc, self.nodes, inicio.elapsed().as_millis() as u64);

            if mejor_sc.abs() >= MATE - 1000 {
                break;
            }
            if let Some(ms) = movetime_ms {
                if inicio.elapsed().as_millis() as u64 > ms * 45 / 100 {
                    break;
                }
            }
        }
        // La raiz nunca pasa por negamax (el loop de arriba la maneja
        // aparte), asi que sin esto la TT no tiene entrada para ella y
        // extraer_pv() no puede ni arrancar a caminarla. Guardarla aca no
        // afecta la busqueda en si (pasa DESPUES del loop).
        if let Some(mv) = mejor_mv {
            self.tt_store(b.zobrist, mejor_prof, mejor_sc, TTFlag::Exact, Some(mv));
        }
        (mejor_mv, mejor_sc, mejor_prof)
    }
}

// ============================================================
//  Lazy SMP: varios hilos nativos buscando la misma posicion raiz en
//  paralelo, compartiendo la TT (con locks por casillero). Cada hilo tiene
//  su propio killers/history (no compartidos, no hay beneficio claro y
//  complica el codigo sin necesidad). El resultado final es el del hilo
//  que llego mas profundo (o, empatados, el de score mas decisivo).
// ============================================================

pub struct ResultadoHilo {
    pub mv: Option<Move>,
    pub score: i32,
    pub profundidad: i32,
    pub nodos: u64,
}

/// Busca `b` con `n_hilos` hilos nativos compartiendo TT, con el mismo
/// presupuesto de reloj que una busqueda de un solo hilo (el paralelismo es
/// para ver MAS nodos en el mismo tiempo, no para tardar mas). Variacion
/// entre hilos: los hilos de indice impar arrancan con las dos primeras
/// jugadas del orden intercambiadas, para que no todos exploren exactamente
/// la misma linea primero -- ademas de la variacion natural que ya aporta
/// el timing real de acceso a la TT compartida entre hilos genuinamente
/// concurrentes (el mecanismo clasico detras de Lazy SMP).
/// `tt` se pasa ya construida (y se espera que el LLAMADOR la guarde y
/// reutilice entre jugadas de la misma partida, igual que la TT de un
/// Searcher normal persiste entre llamadas a "go" -- si se reconstruyera
/// de cero en cada jugada, Lazy SMP perderia la continuidad de la TT entre
/// plies, una desventaja injusta frente a la version de un solo hilo.
pub fn buscar_lazy_smp(
    b: &Board,
    movetime_ms: Option<u64>,
    max_depth: i32,
    n_hilos: usize,
    tt: &Arc<SharedTT>,
    tt_mask: usize,
    modo_lmr: bool,
    game_history: &[u64],
    external_stop: Arc<AtomicBool>,
) -> (Option<Move>, i32, u64, Vec<ResultadoHilo>) {
    if n_hilos <= 1 {
        let mut s = Searcher::new_con_tt_compartida(Arc::clone(tt), tt_mask, modo_lmr);
        s.set_external_stop(Some(external_stop));
        s.set_game_history(game_history.to_vec());
        let (mv, sc, prof) = s.search_time(b, movetime_ms, max_depth, |_, _, _, _| {});
        let nodos = s.nodes;
        return (mv, sc, nodos, vec![ResultadoHilo { mv, score: sc, profundidad: prof, nodos }]);
    }

    let board_copy = *b;

    let handles: Vec<_> = (0..n_hilos)
        .map(|i| {
            let external_stop = Arc::clone(&external_stop);
            let tt = Arc::clone(tt);
            let game_history = game_history.to_vec();
            std::thread::spawn(move || {
                let mut s = Searcher::new_con_tt_compartida(tt, tt_mask, modo_lmr);
                s.variante_orden_raiz = i % 2 == 1;
                s.set_external_stop(Some(external_stop));
                s.set_game_history(game_history);
                let (mv, sc, prof) = s.search_time(&board_copy, movetime_ms, max_depth, |_, _, _, _| {});
                ResultadoHilo { mv, score: sc, profundidad: prof, nodos: s.nodes }
            })
        })
        .collect();

    let resultados: Vec<ResultadoHilo> = handles.into_iter().map(|h| h.join().expect("hilo de busqueda con panic")).collect();

    let nodos_totales: u64 = resultados.iter().map(|r| r.nodos).sum();
    // v12: NO usar score.abs() para desempatar entre hilos con la misma
    // profundidad. Todos buscan la MISMA posicion raiz con el MISMO bando a
    // mover, asi que un score mas alto es sencillamente mejor -- no hace
    // falta "decision" alguna. Con abs(), un hilo que por suerte de orden de
    // jugadas NO vio una refutacion real (score optimista, ej. +400) le
    // ganaba a otro hilo que SI la encontro (score correcto pero cauteloso,
    // ej. -50), porque |400| > |-50| -- eligiendo la evaluacion equivocada
    // con mas confianza en vez de la correcta. Score crudo (sin abs) elige
    // siempre la mejor evaluacion real entre los hilos empatados en profundidad.
    let mejor = resultados
        .iter()
        .max_by_key(|r| (r.profundidad, r.score))
        .expect("al menos un hilo");

    (mejor.mv, mejor.score, nodos_totales, resultados)
}
