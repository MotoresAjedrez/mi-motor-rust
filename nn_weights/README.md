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

## Estado de rendimiento (medido, v13.1)

Primera versión (recompute denso de las 770 entradas en cada llamada):
~10,000-12,000 nodos/seg con `UseNN` activado, más de 100x más lento que
sin la red (~2,000,000 nodos/seg) -- impracticable para juego real.

**Optimización aplicada (v13.1)**: la entrada de la red es un one-hot
disperso -- de las 770 entradas, solo ~32-34 valen 1.0 (una por pieza en
el tablero, más los bits de turno/enroque), el resto son ceros que no
aportan nada. En vez de recorrer las 770 entradas, el forward pass ahora
parte de los sesgos y SUMA solo la columna de pesos de cada entrada
activa (matemáticamente idéntico al cálculo denso -- verificado
comparando el score exacto en cada profundidad antes/después, coinciden
bit a bit). Además, la matriz de la primera capa se guarda transpuesta
(columna-mayor) para que cada suma sea un acceso contiguo a memoria, y
el forward pass no reserva memoria dinámica por llamada (arreglos de
tamaño fijo en la pila, no `Vec`).

**Resultado medido**: ~290,000-330,000 nodos/seg con `UseNN` activado --
de >100x más lento a **~6-7x más lento** que sin la red. Sigue siendo un
costo real (algo así como 1.5-2 plies menos de profundidad efectiva en
el mismo tiempo), pero ya es un trade-off razonable a evaluar, no algo
impracticable. Sin `UseNN` activado, el costo sigue siendo CERO (medido:
~2,000,000-2,400,000 nodos/seg, idéntico a sin la red).

**Nota**: esto sigue sin ser NNUE real -- un accumulator incremental de
verdad (actualizar solo el efecto de la pieza que se movió jugada a
jugada, sin recalcular ni siquiera las ~32 entradas activas desde cero
en cada nodo) daría otra mejora sustancial encima de esta, pero requiere
tocar la recursión de `negamax` en `search.rs` para llevar el estado del
accumulator ply por ply -- un cambio de mayor riesgo/alcance que se dejó
fuera de esta pasada a propósito, dado lo sensible que es esa función.
