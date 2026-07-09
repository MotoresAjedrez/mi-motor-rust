# Pesos de la red neuronal ligera (v13)

`pesos_v1.bin` son los pesos de `~/mi-motor/red_entrenada/pesos.npz`
(motor Python, arquitectura 770→256(ReLU)→32(ReLU)→1) exportados a un
binario plano (f32 little-endian, sin encabezado) en el orden:
`W1(256x770), b1(256), W2(32x256), b2(32), W3(1x32), b3(1)`.

## Cómo se generó

```python
import numpy as np
d = np.load("~/mi-motor/red_entrenada/pesos.npz")
with open("pesos_v1.bin", "wb") as f:
    for k in ["W1", "b1", "W2", "b2", "W3", "b3"]:
        f.write(d[k].astype("<f4").tobytes(order="C"))
```

## Cómo reentrenar con más datos (PGN/FEN)

1. Generar posiciones etiquetadas: `~/mi-motor/generar_datos.py` descarga
   partidas reales (Lichess), muestrea posiciones no forzadas, y las
   etiqueta con Stockfish usando `score.pov(board.turn)` (evaluación desde
   la perspectiva del bando que mueve -- **crítico**: si se reentrena con
   otra convención de signo, `neural.rs` va a interpretar los números al
   revés sin ningún error de compilación que lo avise).
2. Entrenar: `~/mi-motor/entrenar_red.py` (PyTorch, arquitectura fija
   770→256→32→1, labels en `cp/100.0`).
3. Exportar con el script de arriba a un nuevo `pesos_vN.bin`.
4. Cargar en el motor Rust: `setoption name NNPath value nn_weights/pesos_vN.bin`
   seguido de `setoption name UseNN value true`.

## Encoding de entrada (770 floats)

Debe coincidir EXACTO con `~/mi-motor/features_red.py::board_a_vector` y
con `neural.rs::vector_entrada`:
- 0..383: piezas blancas, 6 bloques de 64 casilleros en orden
  Peón/Caballo/Alfil/Torre/Dama/Rey (mismo orden que `ALL_PIECE_TYPES`
  en `types.rs`), casillero = `rank*8+file` (a1=0, h8=63, igual que
  python-chess).
- 384..767: mismo esquema para piezas negras.
- 768: 1.0 si mueven las blancas, 0.0 si mueven las negras.
- 769: 1.0 si el bando que mueve conserva algún derecho de enroque.

## Estado de rendimiento (medido, v13)

**Advertencia honesta, no un "por si acaso":** con esta arquitectura (recompute
completo en cada llamada a `evaluate()`, sin accumulator incremental tipo
NNUE real) la velocidad de búsqueda cae de ~2,000,000 nodos/seg a
~10,000-12,000 nodos/seg cuando `UseNN` está activado -- más de 100 veces
más lento. Por eso queda **apagada por defecto** y no se recomienda
activarla en juego real todavía: la pérdida de profundidad de búsqueda
probablemente pesa mucho más que cualquier mejora de precisión de la red.
Para que esto sea usable en partidas reales hace falta un accumulator
incremental de verdad (actualizar solo el efecto de la pieza que se movió
en cada jugada, no recalcular las 770 entradas desde cero) -- eso es
trabajo de una sesión aparte, no un ajuste rápido.
