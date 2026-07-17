# MiMotor Tal

Ficha para listados de motores (estilo CCRL) y uso en GUIs de ajedrez
(Cutechess, Arena, BanksiaGUI, etc.). Para documentacion tecnica completa
del codigo ver [README.md](README.md).

| Campo | Valor |
|---|---|
| Nombre | MiMotor Tal |
| Autor | Tavito |
| Version | 0.8.0 |
| Lenguaje | Rust |
| Protocolo | UCI |
| Licencia | Codigo fuente disponible en este repositorio |
| Sistema operativo | Windows (x86_64), Linux (x86_64), macOS (Apple Silicon) |
| Hilos | Si (Lazy SMP, opcion `Threads`) |
| Tablas de finales | Syzygy (opcion `SyzygyPath`) |
| Libro de aperturas | Polyglot (opciones `BookPath` / `OwnBook`) |
| Evaluacion | Hibrida: red neuronal NNUE + evaluacion clasica |

## Descripcion

MiMotor Tal es un motor de ajedrez UCI escrito en Rust. Usa una red
neuronal NNUE propia (5378 entradas: piece-square clasico + features de
"amenazas", que capturan que pieza ataca a cual) combinada con una
evaluacion clasica, busqueda alfa-beta con Lazy SMP multi-hilo,
transposition table, quiescence search, y las reducciones/podas
habituales (LMR, null move, futility, SEE). El proyecto completo,
incluida la red neuronal, fue entrenado y ajustado con pruebas h2h
(head-to-head, binario candidato vs binario de referencia) antes de
cada cambio -- ver el historial de mejoras en [README.md](README.md).

## Binarios precompilados

En la seccion "Releases" de este repositorio en GitHub hay binarios
listos para:

- Windows x86_64 (generico, sin instrucciones AVX2 especificas)
- Linux x86_64 (generico, sin instrucciones AVX2 especificas)
- macOS Apple Silicon (arm64, con ruta SIMD NEON dedicada)

Los binarios de Windows y Linux se generaron por cross-compilation
desde macOS y se verifico que compilan sin errores y tienen el formato
ejecutable correcto, pero no se probaron corriendo en una maquina
Windows/Linux real. La logica del motor es identica a la version que
juega en Lichess (bot **chatgpt5_5**), que si esta validada
extensivamente.

## Compilar desde el codigo fuente

```bash
git clone https://github.com/MotoresAjedrez/mi-motor-rust.git
cd mi-motor-rust
cargo build --release
```

Requiere una version reciente de Rust (edition 2024). El ejecutable
queda en `target/release/mi-motor-rust`.

## Configuracion recomendada para pruebas/torneos

```text
setoption name Hash value 128
setoption name Threads value 4
setoption name Move Overhead value 75
setoption name NNUEPath value <ruta a pesos_amenazas_prueba.bin>
setoption name UseNNUE value true
setoption name BookPath value <ruta a books/performance.bin>
setoption name OwnBook value true
```

## Opciones UCI

| Opcion | Tipo | Default | Descripcion |
|---|---|---|---|
| `Hash` | spin | 64 | Memoria de la transposition table en MiB (1-1024). |
| `Threads` | spin | 1 | Hilos de busqueda Lazy SMP (1-16). |
| `Move Overhead` | spin | 75 | Margen de milisegundos por jugada para latencia de GUI/red. |
| `Clear Hash` | button | -- | Vacia la transposition table. |
| `NNUEPath` | string | vacio | Ruta al archivo de pesos NNUE actual. |
| `UseNNUE` | check | false | Activa la evaluacion hibrida NNUE + clasica. |
| `QSearchNNUE` | check | true | Usa NNUE tambien dentro de quiescence search. |
| `NNUEClassicalDepth` | spin | 0 | Profundidad desde el horizonte en la que se usa solo evaluacion clasica (0-4). |
| `SyzygyPath` | string | vacio | Ruta a tablas de finales Syzygy. |
| `BookPath` | string | vacio | Ruta a libro de aperturas Polyglot. |
| `OwnBook` | check | true | Permite que el motor use su propio libro. |
| `Personalidad` | combo | tal | Perfil de estilo de juego (`tal` / `universal`). |

## Comandos utiles

- `bench [profundidad]`: benchmark estandar sobre 6 posiciones fijas,
  imprime un resumen parseable (nodos totales, tiempo, nodos/segundo),
  igual que el comando `bench` de Stockfish/OpenBench.
- `go depth N`: busqueda a profundidad fija, con reporte `info` por
  iteracion (depth/score/nodes/time/nps), igual que `go movetime`.

## Historial de version

Ver [README.md](README.md), seccion "Historial de mejoras", para el
detalle de cada optimizacion aplicada y su resultado en pruebas h2h.
