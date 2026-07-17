# MiMotor Tal

Motor de ajedrez UCI escrito en Rust, con evaluacion hibrida NNUE + clasica.
Juega como bot en Lichess bajo el nombre **chatgpt5_5**.

## Compilar en macOS

Requiere Rust con soporte para edition 2024.

```bash
git clone https://github.com/MotoresAjedrez/mi-motor-rust.git
cd mi-motor-rust
cargo build --release
```

El ejecutable queda en:

```text
target/release/mi-motor-rust
```

Pruebas antes de usarlo:

```bash
cargo test
cargo run --release -- perft
```

## Uso UCI

Opciones principales:

- `Hash`: memoria de la tabla de transposicion en MiB.
- `Threads`: hilos de Lazy SMP.
- `Move Overhead`: margen en milisegundos para GUI/red/sistema operativo.
- `NNUEPath`: ruta al archivo de pesos NNUE (arquitectura actual:
  `pesos_amenazas_prueba.bin`, 5378 entradas -- ver mas abajo).
- `UseNNUE`: activa la evaluacion hibrida despues de cargar pesos validos.
- `QSearchNNUE`: usa NNUE tambien dentro de quiescence (activado por
  defecto; probado a desactivarlo para ganar velocidad y perdio fuerza
  real en h2h, se mantiene activado).
- `SyzygyPath`, `BookPath`/`OwnBook`: tablas de finales y libro Polyglot.

Ejemplo:

```text
setoption name NNUEPath value pesos_amenazas_prueba.bin
setoption name UseNNUE value true
```

## Arquitectura NNUE

`5378 -> 256 (ReLU) -> 32 (ReLU) -> 1 (lineal)`, entrada dispersa (acumulador
incremental):

- 770 entradas base: piece-square para las 6 piezas x 2 colores x 64
  casillas, mas turno y derechos de enroque.
- 4608 entradas de "amenazas": quien ataca a cual pieza (2 colores
  atacantes x 6 tipos de pieza atacante x 6 tipos de pieza victima x 64
  casillas).

El acumulador se actualiza de forma incremental para AMBAS partes: la base
(piece-square, como cualquier NNUE clasica) y las amenazas, que son mas
delicadas porque pueden cambiar para piezas que no se movieron (una jugada
puede abrir o cerrar una linea de ataque de otra pieza). La actualizacion
detecta que casillas cambiaron, que piezas deslizantes quedan en linea con
esas casillas, y ajusta solo eso -- evita recalcular las 5378 features en
cada jugada. Verificado con un fuzz test de ~1920 posiciones aleatorias
comparando el resultado incremental contra el recalculo completo (ver
`src/neural.rs`, test `acumulador_incremental_amenazas_fuzz_determinista`).

## Archivos binarios

- `pesos_amenazas_prueba.bin`: pesos de la arquitectura NNUE ACTUAL en
  produccion (5378 entradas, ver arriba). Es el que hay que usar en
  `NNUEPath` para jugar con la fuerza real del motor.
- `pesos_v1.bin`: fixture de una arquitectura vieja (770 entradas, sin
  amenazas), conservado solo porque dos tests unitarios en `src/neural.rs`
  (`rechaza_nan_sin_panico`, `checksum_es_estable`) lo usan via
  `include_bytes!` para probar el validador de bytes -- no representa la
  red que juega hoy.
- `books/performance.bin`: libro de aperturas Polyglot.

## Historial de mejoras (sesiones recientes)

- **Actualizacion incremental de amenazas**: recupera ~40% de nodos/seg y
  +1 ply de profundidad frente al recalculo completo (h2h 59.3%/150
  partidas).
- **Optimizaciones de velocidad pura** (sin cambiar decisiones de
  busqueda): menos recalculo de SEE al ordenar, menos allocaciones en el
  actualizador de amenazas, cachés clasicas que no se rehacen de mas.
  Verificado bit-identico a profundidad fija antes de medir h2h (70.3%/150
  partidas).
- **badcap**: capturas de SEE negativo reordenadas al final del orden de
  jugadas (h2h 61.3%).
- **quant2**: cuantizacion entera (i8/i16) de la capa L1->L2 de la NNUE,
  +25.7% de velocidad real (h2h 59.5%/100 partidas a 600ms).
- **LMR2**: Late Move Reduction mas agresivo (h2h 58.6%/250 partidas).

Cada cambio se midio de forma aislada (binario candidato vs binario
desplegado, mismos pesos) antes de desplegarse; los candidatos que no
superaron el umbral de fuerza (>55% de puntos en h2h) se descartaron y no
estan en este historial.

## Estado de validacion

`cargo test` corre 31/31 pruebas, incluidos tests de perft, mate, SEE,
repeticion, y los fuzz de acumuladores incrementales (clasico y de
amenazas) que comparan contra el recalculo completo en posiciones
aleatorias.
