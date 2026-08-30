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

    # Menor angulo interno (graus) tolerado no poligono aproximado do
    # contorno. Complementa a circularidade: uma forma pode ter area e
    # circularidade dentro da faixa e ainda ter uma ponta saindo dela (ex.:
    # a estrela/triangulo interno "vazando" um bico na borda do circulo
    # branco depois do CLOSE). Comece tolerante (angulos pequenos so' de
    # verdade sao pontas obvias) e aperte se estiver deixando lixo passar.
    min_vertex_angle_deg: float = 30.0

    # Um quadrado tem circularidade ~0.785 e cantos de 90 -- passa fácil nos
    # dois filtros acima (pensados para pegar ESTRELA/TRIANGULO, nao
    # QUADRADO), vira candidato aceito, e o Hough so' ajusta um circulo
    # inscrito nele. Rejeita explicitamente quem aproxima para exatamente 4
    # vertices (contorno cru OU hull), antes mesmo de olhar circularidade.
    reject_quadrilaterals: bool = True

    # Um circulo com uma pequena falha na borda (a mascara nao fechou 100%
    # num ponto -- reflexo, sombra, CLOSE insuficiente) vira um contorno com
    # uma reentrancia: circularidade e angulo despencam mesmo a base sendo,
    # na pratica, "quase um circulo fechado". Antes de descartar por forma,
    # tenta o HULL CONVEXO do mesmo contorno -- o hull ignora reentrancias
    # pequenas (e' a menor forma convexa que contem os pontos), entao fecha
    # esse tipo de falha sem precisar de mais CLOSE (que arrisca fundir
    # bases vizinhas). So' e' usado se o contorno CRU falhar primeiro.
    use_hull_fallback: bool = True

    # ── Refino por Hough (cv2.HoughCircles) ──────────────────────────────
    # O contorno que sobrevive aos filtros acima raramente e' um circulo
    # perfeito -- CLOSE/OPEN e a forma preta interna deixam a borda
    # irregular, e minEnclosingCircle superestima o raio para se ajustar ao
    # ponto mais saliente. Rodar Hough SO' no recorte ao redor de cada
    # contorno aceito (nao na imagem inteira) acha o circulo que melhor
    # explica aquele blob, sem o custo/risco de falsos positivos de rodar
    # Hough na imagem toda.
    use_hough_refine: bool = True
    hough_dp: float = 1.0            # resolucao do acumulador (1.0 = igual a imagem)
    hough_param1: float = 80.0       # limiar alto do Canny interno do Hough
    hough_param2: float = 20.0       # limiar do acumulador -- menor acha mais circulos
    # Raio buscado = raio do contorno +- esta fracao (0.3 = -30% a +30%).
    hough_radius_margin: float = 0.3

    # ── Blur pre-segmentacao ──────────────────────────────────────────────
    # Median blur no HSV antes do threshold -- reduz ruido tipo sal-e-pimenta
    # (reflexo pontual, textura da lona) que sobrevive ao CLOSE/OPEN como
    # furo/ilha isolada na mascara. median_blur_ksize precisa ser impar; 1
    # equivale a nao borrar.
    use_median_blur: bool = True
    median_blur_ksize: int = 5


def white_mask(frame: np.ndarray, params: DetectionParams) -> np.ndarray:
    """Mascara binaria do branco (saturacao baixa, valor alto) + CLOSE/OPEN."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if params.use_median_blur and params.median_blur_ksize > 1:
        hsv = cv2.medianBlur(hsv, int(params.median_blur_ksize))

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


def _aproximar_poligono(contour) -> np.ndarray:
    """Poligono aproximado do contorno (approxPolyDP, epsilon = 2% do
    perimetro -- o valor classico: fino o bastante para nao arredondar
    pontas/cantos reais, grosso o bastante para nao ver serrilhado de pixel
    como vertice). Usado tanto para o angulo minimo quanto para excluir
    quadrilateros (ver _eh_quadrilatero) -- uma unica aproximacao para as
    duas checagens, em vez de recalcular approxPolyDP duas vezes."""
    perimeter = cv2.arcLength(contour, closed=True)
    if perimeter <= 0:
        return np.empty((0, 2))
    epsilon = 0.02 * perimeter
    return cv2.approxPolyDP(contour, epsilon, closed=True).reshape(-1, 2)


def _eh_quadrilatero(approx: np.ndarray) -> bool:
    """Um quadrado tem circularidade ~0.785 (4*pi*area/perimetro^2 de um
    quadrado e' exatamente pi/4) e cantos de 90 -- passa fácil em
    min_circularity e min_vertex_angle_deg default, que foram pensados para
    filtrar ESTRELA/TRIANGULO, nao QUADRADO. Sem essa checagem explicita de
    vertices, um quadrado vira candidato aceito e o Hough so' ajusta um
    circulo inscrito nele -- o sintoma vira "Hough detectando coisa
    quadrada", mas a causa e' o filtro de forma deixando passar antes."""
    return len(approx) == 4


def _min_vertex_angle_deg(approx: np.ndarray) -> float:
    """Menor angulo interno do poligono aproximado `approx`, em graus.

    Um circulo aproxima para um poligono de muitos lados com angulos
    obtusos (perto de 180 deg); uma ponta de estrela/triangulo aproxima para
    um angulo agudo bem marcado.

    Devolve 180.0 (nunca rejeita) se o poligono tiver menos de 3 vertices
    -- sem poligono fechado, nao ha angulo para medir."""
    n = len(approx)
    if n < 3:
        return 180.0

    menor = 180.0
    for i in range(n):
        anterior = approx[i - 1].astype(np.float64)
        atual = approx[i].astype(np.float64)
        proximo = approx[(i + 1) % n].astype(np.float64)

        v1 = anterior - atual
        v2 = proximo - atual
        norma = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norma <= 1e-9:
            continue

        cos_ang = np.clip(np.dot(v1, v2) / norma, -1.0, 1.0)
        angulo = np.degrees(np.arccos(cos_ang))
        menor = min(menor, angulo)

    return menor


def _refinar_com_hough(mask: np.ndarray, cx: float, cy: float, radius: float,
                        params: DetectionParams):
    """Roda HoughCircles num recorte ao redor de (cx, cy, radius) -- nao na
    imagem inteira -- para achar o circulo que melhor explica aquele blob
    especifico. Devolve (cx, cy, radius, encontrado); se Hough nao achar
    nada no recorte, devolve o (cx, cy, radius) originais (do
    minEnclosingCircle) sem alterar, e encontrado=False.
    """
    if radius <= 0:
        return cx, cy, radius, False

    pad = int(radius * 0.5) + 5
    x0 = max(0, int(cx - radius) - pad)
    y0 = max(0, int(cy - radius) - pad)
    x1 = min(mask.shape[1], int(cx + radius) + pad)
    y1 = min(mask.shape[0], int(cy + radius) + pad)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return cx, cy, radius, False

    recorte = cv2.GaussianBlur(mask[y0:y1, x0:x1], (5, 5), 0)

    margem = float(params.hough_radius_margin)
    min_r = max(1, int(radius * (1.0 - margem)))
    max_r = max(min_r + 1, int(radius * (1.0 + margem)))

    circles = cv2.HoughCircles(
        recorte, cv2.HOUGH_GRADIENT,
        dp=max(0.1, float(params.hough_dp)),
        minDist=max(recorte.shape),  # um so' circulo esperado no recorte
        param1=max(1.0, float(params.hough_param1)),
        param2=max(1.0, float(params.hough_param2)),
        minRadius=min_r, maxRadius=max_r)

    if circles is None or len(circles[0]) == 0:
        return cx, cy, radius, False

    hx, hy, hr = circles[0][0]
    return x0 + float(hx), y0 + float(hy), float(hr), True


def _forma_valida(circularity: float, vertex_angle: float, params: DetectionParams) -> bool:
    return circularity >= params.min_circularity and vertex_angle >= params.min_vertex_angle_deg


def find_candidates(mask: np.ndarray, frame_shape, params: DetectionParams) -> list:
    """Todos os contornos da mascara, aceitos ou nao, com o motivo da
    rejeicao (None se aceito). Base tanto do filtro de producao
    (find_circles) quanto de ferramentas de debug que precisam mostrar POR
    QUE um contorno foi descartado -- se so' devolvessemos os aceitos, uma
    calibracao ruim pareceria "nao achou nada" sem dizer se o problema foi
    area ou forma.
    """
    img_area = frame_shape[0] * frame_shape[1]
    min_area_px = float(params.min_area_fraction) * img_area
    max_area_px = float(params.max_area_fraction) * img_area
    min_circularity = float(params.min_circularity)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, closed=True)
        circularity = (4.0 * np.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0.0
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        approx = _aproximar_poligono(contour)
        vertex_angle = _min_vertex_angle_deg(approx)
        closed_via_hull = False
        hull_poly = None

        if perimeter <= 0:
            rejected_by = 'perimetro'
        elif area < min_area_px:
            rejected_by = 'area_pequena'
        elif area > max_area_px:
            rejected_by = 'area_grande'
        elif params.reject_quadrilaterals and _eh_quadrilatero(approx):
            rejected_by = 'quadrilatero'
        elif _forma_valida(circularity, vertex_angle, params):
            rejected_by = None
        elif params.use_hull_fallback:
            # O contorno cru falhou em forma -- tenta o hull convexo antes
            # de descartar (ver comentario em DetectionParams.use_hull_fallback).
            hull = cv2.convexHull(contour)
            hull_approx = _aproximar_poligono(hull)
            hull_perimeter = cv2.arcLength(hull, closed=True)
            hull_area = cv2.contourArea(hull)
            hull_circularity = (4.0 * np.pi * hull_area / (hull_perimeter ** 2)
                                 if hull_perimeter > 0 else 0.0)
            hull_vertex_angle = _min_vertex_angle_deg(hull_approx)

            if params.reject_quadrilaterals and _eh_quadrilatero(hull_approx):
                rejected_by = 'quadrilatero'
            elif (hull_perimeter > 0 and hull_area <= max_area_px
                    and _forma_valida(hull_circularity, hull_vertex_angle, params)):
                area, circularity, vertex_angle = hull_area, hull_circularity, hull_vertex_angle
                (cx, cy), radius = cv2.minEnclosingCircle(hull)
                closed_via_hull = True
                hull_poly = hull
                rejected_by = None
            else:
                rejected_by = 'circularidade' if circularity < min_circularity else 'angulo_agudo'
        else:
            rejected_by = 'circularidade' if circularity < min_circularity else 'angulo_agudo'

        # So' vale refinar com Hough quem passou em TODOS os filtros de
        # contorno -- rodar em lixo rejeitado seria custo sem uso, e o
        # (cx, cy, radius) de quem foi rejeitado nem aparece em find_circles.
        hough_ok = False
        if rejected_by is None and params.use_hough_refine:
            cx, cy, radius, hough_ok = _refinar_com_hough(mask, cx, cy, radius, params)

        candidates.append({
            'contour': contour,
            'hull': hull_poly,
            'center_px': (cx, cy),
            'radius': radius,
            'center_norm': (cx / frame_shape[1], cy / frame_shape[0]),
            'area': float(area),
            'circularity': float(circularity),
            'vertex_angle_deg': float(vertex_angle),
            'closed_via_hull': closed_via_hull,
            'hough_refined': hough_ok,
            'rejected_by': rejected_by,
        })

    return candidates


def find_circles(mask: np.ndarray, frame_shape, params: DetectionParams) -> list:
    """Contornos da mascara, filtrados por area e circularidade."""
    detections = []
    for c in find_candidates(mask, frame_shape, params):
        if c['rejected_by'] is not None:
            continue
        detections.append({
            'center_px': c['center_px'],
            'radius': c['radius'],
            'center_norm': c['center_norm'],
            'area': c['area'],
            'circularity': c['circularity'],
        })
    return detections


def detect(frame: np.ndarray, params: DetectionParams):
    """Atalho: mascara + contornos filtrados. Devolve (detections, mask)."""
    mask = white_mask(frame, params)
    return find_circles(mask, frame.shape, params), mask


def detect_debug(frame: np.ndarray, params: DetectionParams):
    """Como detect(), mas tambem devolve TODOS os candidatos (aceitos e
    rejeitados, com motivo) -- para ferramentas de debug/calibracao
    desenharem o que o filtro descartou e por que. Devolve (detections,
    mask, candidates)."""
    mask = white_mask(frame, params)
    candidates = find_candidates(mask, frame.shape, params)
    detections = [
        {k: v for k, v in c.items() if k not in ('contour', 'hull', 'rejected_by')}
        for c in candidates if c['rejected_by'] is None
    ]
    return detections, mask, candidates
