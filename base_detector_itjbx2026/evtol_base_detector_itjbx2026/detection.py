"""
Deteccao do circulo branco -- sem ROS, para o no de verdade (node.py) e a
ferramenta de calibracao (calibrate.py) usarem exatamente a mesma logica.

Duas copias do mesmo algoritmo divergem cedo ou tarde: alguem ajusta um dos
dois e esquece do outro, e a calibracao para de bater com o que o detector
real faz. Uma unica fonte de verdade evita isso -- ver o docstring de
node.py para a explicacao completa da estrategia (branco + CLOSE +
circularidade).
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DetectionParams:
    white_sat_max: int = 60
    white_val_min: int = 180
    close_kernel_size: int = 15
    close_iterations: int = 2
    open_kernel_size: int = 3
    open_iterations: int = 1
    min_circularity: float = 0.75
    min_area_fraction: float = 0.001
    max_area_fraction: float = 0.25


def white_mask(frame: np.ndarray, params: DetectionParams) -> np.ndarray:
    """Mascara binaria do branco (saturacao baixa, valor alto) + CLOSE/OPEN."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 0, int(params.white_val_min)])
    upper = np.array([179, int(params.white_sat_max), 255])
    mask = cv2.inRange(hsv, lower, upper)

    # CLOSE primeiro religa as reentrancias que a forma preta interna abre na
    # borda do circulo; OPEN depois tira ruido pequeno solto. A ordem
    # importa: OPEN antes do CLOSE apagaria reentrancias finas antes de o
    # CLOSE ter chance de religa-las.
    close_size = max(1, int(params.close_kernel_size))
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    close_iters = max(0, int(params.close_iterations))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k, iterations=close_iters)

    open_size = max(1, int(params.open_kernel_size))
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
    open_iters = max(0, int(params.open_iterations))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k, iterations=open_iters)

    return mask


def find_circles(mask: np.ndarray, frame_shape, params: DetectionParams) -> list:
    """Contornos externos da mascara, filtrados por area e circularidade."""
    img_area = frame_shape[0] * frame_shape[1]
    min_area_px = float(params.min_area_fraction) * img_area
    max_area_px = float(params.max_area_fraction) * img_area
    min_circularity = float(params.min_circularity)

    # RETR_EXTERNAL: so o contorno mais externo de cada regiao -- um buraco
    # interno que sobrou do CLOSE e ignorado por definicao, contanto que nao
    # encoste na borda (ver docstring de node.py).
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area_px or area > max_area_px:
            continue

        perimeter = cv2.arcLength(contour, closed=True)
        if perimeter <= 0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        if circularity < min_circularity:
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(contour)

        detections.append({
            'center_px': (cx, cy),
            'radius': radius,
            'center_norm': (cx / frame_shape[1], cy / frame_shape[0]),
            'area': float(area),
            'circularity': float(circularity),
        })

    return detections


def detect(frame: np.ndarray, params: DetectionParams):
    """Atalho: mascara + contornos filtrados. Devolve (detections, mask)."""
    mask = white_mask(frame, params)
    return find_circles(mask, frame.shape, params), mask
