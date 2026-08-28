#!/usr/bin/env python3
"""
Teste de deteccao em camera real, SEM ROS.

Abre a camera USB direto (cv2.VideoCapture), roda a MESMA deteccao do no' de
producao (evtol_base_detector_itjbx2026.detection -- nao uma copia) e imprime
no terminal as coordenadas de cada base encontrada, frame a frame.

Existe porque calibrate.py exige a pilha ROS inteira de pe' (rclpy +
webcam_publisher publicando /vertical_camera/compressed) so' para ver se a
deteccao funciona numa camera nova -- overhead que nao faz sentido quando a
pergunta e' so' "essa camera, nesta luz, acha o circulo branco?".

Uso -- standalone, SEM ROS (sem colcon build, sem source install/setup.bash)
---------------------------------------------------------------------------
    python3 webcam_test.py                          # /dev/video0, YAML padrao
    python3 webcam_test.py --device /dev/video2
    python3 webcam_test.py --device 2 --sem-janela   # so' imprime, sem GUI
    python3 webcam_test.py --config /path/outro.yaml
    python3 webcam_test.py --calibrar                # sliders ao vivo (ex-calibrate.py)

Depende so' de opencv-python, numpy e pyyaml (`pip install opencv-python
numpy pyyaml` se a maquina de teste nao tiver o workspace ROS montado).

'q' na janela (ou Ctrl+C no terminal) encerra. Em --calibrar, ao sair imprime
um bloco pronto para colar em config/flight.yaml.
"""

import argparse
import os
import sys
import time


def _importar_cv2_sem_conflito_numpy():
    """O cv2 do apt (python3-opencv, o que o ROS usa) e compilado contra
    numpy 1.x. Se houver um numpy 2.x mais novo em ~/.local (pip --user),
    ele vem primeiro no sys.path e o import de cv2 quebra com
    "AttributeError: _ARRAY_API not found" -- sintoma de ABI incompativel,
    nao de cv2 ausente. Corrige tirando ~/.local do caminho e tentando de
    novo com os pacotes do sistema, sem desinstalar nem downgradar nada."""
    try:
        import cv2
        return cv2
    except (ImportError, AttributeError):
        local = os.path.expanduser('~/.local')
        sys.path[:] = [p for p in sys.path if not p.startswith(local)]
        for nome in list(sys.modules):
            if nome == 'numpy' or nome.startswith('numpy.') \
                    or nome == 'cv2' or nome.startswith('cv2.'):
                del sys.modules[nome]
        import cv2
        print('aviso: numpy 2.x de ~/.local conflitava com o cv2 do sistema '
              '-- usando os pacotes do apt para este processo', file=sys.stderr)
        return cv2


cv2 = _importar_cv2_sem_conflito_numpy()
import numpy as np  # noqa: E402
import yaml  # noqa: E402

try:
    # Dentro do pacote instalado (ros2 run base_detector_itjbx2026 ...).
    from .detection import DetectionParams, detect_debug
except ImportError:
    # Rodado direto (`python3 webcam_test.py`), sem contexto de pacote --
    # detection.py nao depende de ROS, entao basta achar o arquivo do lado.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from detection import DetectionParams, detect_debug

# Cor (BGR) por motivo de rejeicao -- None (aceito) vem em verde, o resto e'
# por que aquele contorno NAO virou deteccao, para nao ter que adivinhar
# olhando so' a mascara binaria.
_CORES_REJEICAO = {
    None: (0, 255, 0),              # aceito
    'circularidade': (0, 0, 255),   # vermelho -- forma nao bateu
    'area_pequena': (0, 165, 255),  # laranja -- pequeno demais
    'area_grande': (0, 165, 255),   # laranja -- grande demais
    'angulo_agudo': (255, 0, 255),  # magenta -- ponta saindo do contorno
    'quadrilatero': (180, 0, 255),  # roxo -- aproxima para ~4 vertices (quadrado/retangulo)
    'perimetro': (128, 128, 128),   # cinza -- contorno degenerado
}
_COR_CENTRO_ACEITO = (255, 255, 0)  # ciano -- distinto do verde do contorno
_COR_HOUGH = (0, 255, 255)          # amarelo -- circulo refinado por HoughCircles
_COR_HULL = (255, 128, 0)           # azul -- hull convexo que salvou o contorno

_JANELA_CAMERA = 'camera (q para sair)'
_JANELA_MASCARA = 'mascara'
_JANELA_CONTROLES = 'controles'

_PARAMS_FIELDS = (
    'white_sat_max', 'white_val_min',
    'close_kernel_size', 'close_iterations',
    'open_kernel_size', 'open_iterations',
    'min_circularity', 'min_area_fraction', 'max_area_fraction',
    'min_vertex_angle_deg', 'reject_quadrilaterals', 'use_hull_fallback',
    'use_hough_refine', 'hough_dp', 'hough_param1', 'hough_param2',
    'hough_radius_margin',
)

_YAML_PADRAO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'config', 'flight.yaml')


def _params_do_yaml(caminho: str) -> DetectionParams:
    """Le os mesmos campos que node.py le, do MESMO arquivo de config -- para
    que este teste use a calibracao de verdade, e nao os defaults do
    dataclass."""
    with open(caminho) as f:
        conteudo = yaml.safe_load(f)

    raiz = conteudo.get('base_detector_itjbx2026', conteudo)
    parametros = raiz.get('ros__parameters', raiz)

    kwargs = {k: parametros[k] for k in _PARAMS_FIELDS if k in parametros}
    return DetectionParams(**kwargs)


def _abrir_camera(device: str, width: int, height: int):
    # Indice numerico ("0", "2") ou caminho de device ("/dev/video2") -- os
    # dois jeitos que se costuma identificar uma USB cam no Linux.
    fonte = int(device) if device.isdigit() else device

    cap = cv2.VideoCapture(fonte, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f'nao foi possivel abrir a camera em {device!r}')

    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return cap


def _make_trackbars(params: DetectionParams, frame_area: int) -> None:
    """Sliders numa janela PROPRIA, separada da mascara -- com 17 sliders,
    misturar com a imagem deixava a janela toda desproporcional (a imagem
    espremida ou os sliders cortados, dependendo do backend do OpenCV).
    min_area_px/max_area_px sao em PIXELS (nao fracao) porque e' o que da
    pra calibrar olhando o tamanho da base na imagem; convertem pra
    min_area_fraction/max_area_fraction na leitura, usando a area do frame
    aberto."""
    cv2.namedWindow(_JANELA_CONTROLES, cv2.WINDOW_NORMAL)
    w = _JANELA_CONTROLES
    noop = lambda _v: None  # noqa: E731

    # Canvas fina so' pra janela ter algo pra mostrar -- o HighGUI cresce a
    # janela pra caber os 17 sliders acima dela de qualquer forma; a imagem
    # em si nao precisa ser grande.
    cv2.imshow(w, np.zeros((10, 420, 3), dtype=np.uint8))
    cv2.resizeWindow(w, 420, 620)

    # OpenCV trackbars nao tem valor minimo != 0, entao os que precisam
    # comecar em 1 (kernels e iteracoes) sao lidos com +1 na leitura.
    cv2.createTrackbar('white_sat_max', w, params.white_sat_max, 255, noop)
    cv2.createTrackbar('white_val_min', w, params.white_val_min, 255, noop)
    cv2.createTrackbar('close_kernel-1', w, params.close_kernel_size - 1, 60, noop)
    cv2.createTrackbar('close_iters', w, params.close_iterations, 5, noop)
    cv2.createTrackbar('open_kernel-1', w, params.open_kernel_size - 1, 30, noop)
    cv2.createTrackbar('open_iters', w, params.open_iterations, 5, noop)
    cv2.createTrackbar('min_circularity_%', w, int(params.min_circularity * 100), 100, noop)
    cv2.createTrackbar('min_vertex_angle', w, int(params.min_vertex_angle_deg), 180, noop)

    # Quadrado tem circularidade ~0.785 e cantos de 90 -- passa nos dois
    # sliders acima sem esforco. Isso rejeita explicitamente quem aproxima
    # para ~4 vertices, antes mesmo de olhar circularidade/angulo.
    cv2.createTrackbar('rejeitar_quad', w, int(params.reject_quadrilaterals), 1, noop)

    min_area_px_atual = int(params.min_area_fraction * frame_area)
    cv2.createTrackbar('min_area_px', w, min_area_px_atual, frame_area, noop)
    max_area_px_atual = int(params.max_area_fraction * frame_area)
    cv2.createTrackbar('max_area_px', w, max_area_px_atual, frame_area, noop)

    # Hull convexo salva contorno com reentrancia (mascara que nao fechou
    # 100% num ponto) que falhou em circularidade/angulo -- ver
    # DetectionParams.use_hull_fallback. Checkbox: 0 desliga, 1 liga.
    cv2.createTrackbar('hull_fallback', w, int(params.use_hull_fallback), 1, noop)

    # Refino por HoughCircles -- roda so' no recorte de quem ja passou nos
    # filtros acima (ver detection.py). 'hough_ativo' funciona como
    # checkbox: 0 desliga, 1 liga.
    cv2.createTrackbar('hough_ativo', w, int(params.use_hough_refine), 1, noop)
    cv2.createTrackbar('hough_dp_x10', w, int(params.hough_dp * 10), 30, noop)
    cv2.createTrackbar('hough_param1', w, int(params.hough_param1), 300, noop)
    cv2.createTrackbar('hough_param2', w, int(params.hough_param2), 100, noop)
    cv2.createTrackbar('hough_margem_%', w, int(params.hough_radius_margin * 100), 100, noop)


def _read_trackbars(frame_area: int) -> DetectionParams:
    def g(name: str) -> int:
        return cv2.getTrackbarPos(name, _JANELA_CONTROLES)

    min_area_fraction = (g('min_area_px') / frame_area) if frame_area > 0 else 0.0
    max_area_fraction = (g('max_area_px') / frame_area) if frame_area > 0 else 1.0

    return DetectionParams(
        white_sat_max=g('white_sat_max'),
        white_val_min=g('white_val_min'),
        close_kernel_size=g('close_kernel-1') + 1,
        close_iterations=g('close_iters'),
        open_kernel_size=g('open_kernel-1') + 1,
        open_iterations=g('open_iters'),
        min_circularity=g('min_circularity_%') / 100.0,
        min_vertex_angle_deg=float(g('min_vertex_angle')),
        min_area_fraction=min_area_fraction,
        max_area_fraction=max_area_fraction,
        reject_quadrilaterals=bool(g('rejeitar_quad')),
        use_hull_fallback=bool(g('hull_fallback')),
        use_hough_refine=bool(g('hough_ativo')),
        hough_dp=max(0.1, g('hough_dp_x10') / 10.0),
        hough_param1=float(max(1, g('hough_param1'))),
        hough_param2=float(max(1, g('hough_param2'))),
        hough_radius_margin=g('hough_margem_%') / 100.0,
    )


def _print_yaml(params: DetectionParams) -> None:
    print('\n# ---- cole em config/flight.yaml ----')
    print(f'    white_sat_max: {params.white_sat_max}')
    print(f'    white_val_min: {params.white_val_min}')
    print(f'    close_kernel_size: {params.close_kernel_size}')
    print(f'    close_iterations: {params.close_iterations}')
    print(f'    open_kernel_size: {params.open_kernel_size}')
    print(f'    open_iterations: {params.open_iterations}')
    print(f'    min_circularity: {params.min_circularity:.2f}')
    print(f'    min_vertex_angle_deg: {params.min_vertex_angle_deg:.1f}')
    print(f'    reject_quadrilaterals: {str(params.reject_quadrilaterals).lower()}')
    print(f'    min_area_fraction: {params.min_area_fraction:.5f}')
    print(f'    max_area_fraction: {params.max_area_fraction:.5f}')
    print(f'    use_hull_fallback: {str(params.use_hull_fallback).lower()}')
    print(f'    use_hough_refine: {str(params.use_hough_refine).lower()}')
    print(f'    hough_dp: {params.hough_dp:.1f}')
    print(f'    hough_param1: {params.hough_param1:.0f}')
    print(f'    hough_param2: {params.hough_param2:.0f}')
    print(f'    hough_radius_margin: {params.hough_radius_margin:.2f}')
    print('# -------------------------------------------------------\n')


def _marcar_centro(img, cx: int, cy: int, cor: tuple) -> None:
    """Cruz + ponto solido no centro exato da deteccao -- o contorno do
    circulo mostra o tamanho, mas nao deixa obvio ONDE o algoritmo marcou o
    centro (o que importa de verdade, ja que e' isso que vira center_norm)."""
    cv2.drawMarker(img, (cx, cy), cor, markerType=cv2.MARKER_CROSS,
                    markerSize=14, thickness=2)
    cv2.circle(img, (cx, cy), 3, cor, -1)


def _desenhar_contornos(img, candidates: list) -> None:
    """Desenha TODOS os contornos avaliados (aceitos ou nao), coloridos pelo
    motivo de rejeicao, com a circularidade de cada um -- responde "por que
    aquele blob nao virou deteccao?" sem precisar adivinhar olhando so' a
    mascara final. Quem foi refinado por Hough ganha tambem o circulo
    amarelo (o que de fato vira center_px/radius), para comparar contra o
    contorno irregular por baixo."""
    for c in candidates:
        cor = _CORES_REJEICAO.get(c['rejected_by'], (0, 0, 255))
        cv2.drawContours(img, [c['contour']], -1, cor, 2)

        cx, cy = int(c['center_px'][0]), int(c['center_px'][1])
        # circularidade e menor angulo interno (deg) -- os dois criterios de
        # forma, lado a lado, para ver de cara qual deles rejeitou.
        cv2.putText(img, f"c={c['circularity']:.2f} a={c['vertex_angle_deg']:.0f}",
                    (cx - 35, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cor, 1)

        if c['rejected_by'] is None:
            if c.get('closed_via_hull') and c.get('hull') is not None:
                cv2.drawContours(img, [c['hull']], -1, _COR_HULL, 1)
            if c.get('hough_refined'):
                cv2.circle(img, (cx, cy), int(c['radius']), _COR_HOUGH, 1)
            _marcar_centro(img, cx, cy, _COR_CENTRO_ACEITO)


def _desenhar_anotado(frame, candidates: list):
    aceitos = sum(1 for c in candidates if c['rejected_by'] is None)
    annotated = frame.copy()
    _desenhar_contornos(annotated, candidates)
    cv2.putText(annotated, f'Bases: {aceitos} (contornos: {len(candidates)})',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return annotated


def _desenhar_mascara_anotada(mask, candidates: list):
    """Mascara binaria em BGR com todos os contornos avaliados por cima --
    verde = virou deteccao, vermelho/laranja/cinza = motivo da rejeicao."""
    anotada = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    _desenhar_contornos(anotada, candidates)
    return anotada


def _rodar_calibracao(cap, params_iniciais: DetectionParams) -> None:
    """Sliders ao vivo sobre a camera aberta direto (sem ROS, sem Detector).
    Mesma UX de calibrate.py: ajuste ate a mascara virar um disco branco
    solido nas bases, 'q' fecha e imprime o YAML pronto para colar."""
    ok, frame = cap.read()
    if not ok:
        print('falha ao ler frame da camera -- device desconectado?')
        return
    frame_area = frame.shape[0] * frame.shape[1]

    _make_trackbars(params_iniciais, frame_area)
    last_params = params_iniciais

    try:
        while True:
            last_params = _read_trackbars(frame_area)
            _detections, mask, candidates = detect_debug(frame, last_params)

            # Contorno pequeno demais nem aparece no debug -- normalmente e'
            # ruido (poeira, textura), e some do desenho conforme o slider
            # min_area_px sobe, em vez de continuar poluindo a tela em
            # laranja como "rejeitado".
            visiveis = [c for c in candidates if c['rejected_by'] != 'area_pequena']

            cv2.imshow(_JANELA_CAMERA, _desenhar_anotado(frame, visiveis))
            cv2.imshow(_JANELA_MASCARA, _desenhar_mascara_anotada(mask, visiveis))

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

            ok, frame = cap.read()
            if not ok:
                print('falha ao ler frame da camera -- device desconectado?')
                break
    except KeyboardInterrupt:
        pass
    finally:
        _print_yaml(last_params)


def _formatar_deteccoes(detections: list, timestamp: float) -> str:
    if not detections:
        return f'[{timestamp:9.3f}s] nenhuma base'

    partes = [f'[{timestamp:9.3f}s] {len(detections)} base(s):']
    for i, det in enumerate(detections):
        cx_px, cy_px = det['center_px']
        cx_n, cy_n = det['center_norm']
        partes.append(
            f'  #{i} px=({cx_px:6.1f}, {cy_px:6.1f})'
            f' norm=({cx_n:.3f}, {cy_n:.3f})'
            f' raio={det["radius"]:5.1f}'
            f' circularidade={det["circularity"]:.2f}')
    return '\n'.join(partes)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--device', default='0',
                     help='indice (0, 1, ...) ou caminho (/dev/video2) da camera USB')
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--config', default=_YAML_PADRAO,
                     help='YAML com os parametros de deteccao (default: o do pacote)')
    ap.add_argument('--sem-janela', action='store_true',
                     help='nao abre janelas do OpenCV, so imprime no terminal')
    ap.add_argument('--calibrar', action='store_true',
                     help='sliders ao vivo para ajustar a segmentacao (ignora --sem-janela); '
                          'ao sair (q / Ctrl+C) imprime o YAML pronto para colar')
    args = ap.parse_args()

    if args.calibrar and args.sem_janela:
        ap.error('--calibrar precisa de janela, nao combina com --sem-janela')

    params = _params_do_yaml(args.config)
    print(f'Parametros lidos de {args.config}:\n  {params}\n')

    cap = _abrir_camera(args.device, args.width, args.height)
    print(f'Camera aberta em {args.device!r} '
          f'({int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x'
          f'{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}). Ctrl+C para sair.\n')

    try:
        if args.calibrar:
            _rodar_calibracao(cap, params)
        else:
            _rodar_deteccao(cap, params, mostrar_janela=not args.sem_janela)
    finally:
        cap.release()
        cv2.destroyAllWindows()


def _rodar_deteccao(cap, params: DetectionParams, mostrar_janela: bool) -> None:
    inicio = time.monotonic()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print('falha ao ler frame da camera -- device desconectado?')
                break

            detections, mask, candidates = detect_debug(frame, params)
            print(_formatar_deteccoes(detections, time.monotonic() - inicio))

            if mostrar_janela:
                cv2.imshow(_JANELA_CAMERA, _desenhar_anotado(frame, candidates))
                cv2.imshow(_JANELA_MASCARA, _desenhar_mascara_anotada(mask, candidates))

                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
