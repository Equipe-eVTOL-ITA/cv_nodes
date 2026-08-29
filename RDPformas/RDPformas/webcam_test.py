        #!/usr/bin/env python3
"""
Teste standalone do RDPformas em camera real, SEM ROS.

RDPformas.py em si NAO foi alterado -- este e' um arquivo NOVO, copiado e
reduzido dele, so' pra testar numa camera USB direto sem precisar da pilha
ROS inteira (rclpy + webcam_publisher). Os filtros/parametros de forma e
ArUco sao os MESMOS valores de RDPformas.py -- se algum dia divergirem, e'
sinal de que um dos dois foi ajustado sem o outro.

O que fica de fora, de proposito (nao faz diferenca pra testar deteccao):
  - Publicacao ROS (BouncingDetection, imagem de debug) -- vira print()
    e janela do OpenCV.
  - Fila assincrona de OCR (_submit_async_ocr/_process_pending_ocr) --
    aqui a leitura do digito e' sincrona, direto no loop; mais simples
    pra um teste manual, sem mudar o RESULTADO da leitura em si.
  - Cache do alvo/divisibilidade (_cached_target_*) -- especifico da
    missao de bouncing, nao ajuda a testar "a forma e o numero estao
    sendo lidos direito?".

Uso -- standalone, SEM ROS (sem colcon build, sem source install/setup.bash)
---------------------------------------------------------------------------
    python3 webcam_test.py                          # /dev/video0
    python3 webcam_test.py --device /dev/video2
    python3 webcam_test.py --device 2 --sem-janela   # so' imprime, sem GUI

Depende de opencv-python (com aruco), numpy, e tesserocr OU pytesseract
(opcionais -- sem nenhum dos dois, cai pra classificacao de digito por
contorno, igual RDPformas.py faz).

'q' na janela (ou Ctrl+C no terminal) encerra.
"""

import argparse
import math
import os
import sys
import threading
from dataclasses import dataclass


def _importar_cv2_sem_conflito_numpy():
    """O cv2 do apt (python3-opencv) e compilado contra numpy 1.x. Se houver
    um numpy 2.x mais novo em ~/.local (pip --user), ele vem primeiro no
    sys.path e o import de cv2 quebra com "AttributeError: _ARRAY_API not
    found" -- sintoma de ABI incompativel, nao de cv2 ausente. Corrige
    tirando ~/.local do caminho e tentando de novo com os pacotes do
    sistema, sem desinstalar nem downgradar nada."""
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
from cv2 import aruco  # noqa: E402

try:
    from tesserocr import PyTessBaseAPI, PSM
    from PIL import Image
    TESSEROCR_AVAILABLE = True
except Exception:
    TESSEROCR_AVAILABLE = False
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except Exception:
    PYTESSERACT_AVAILABLE = False

# Caminhos usuais do tessdata (dados de idioma) por distro/versao do apt.
_TESSDATA_CANDIDATOS = (
    '/usr/share/tesseract-ocr/4.00/tessdata/',  # Ubuntu 22.04 (tesseract-ocr 4.1.1)
    '/usr/share/tesseract-ocr/5/tessdata/',     # Ubuntu 24.04 (tesseract-ocr 5.x)
    '/usr/share/tessdata/',
)


def _achar_tessdata():
    """PyTessBaseAPI() pode falhar com "Failed to init API, possibly an
    invalid tessdata path: ./" mesmo com tesseract-ocr instalado e
    TESSDATA_PREFIX vazio -- acontece quando a lib do Tesseract que o
    tesserocr linkou em tempo de execucao e' de uma instalacao DIFERENTE da
    que o apt colocou os dados de idioma (ex.: tesserocr.tesseract_version()
    reporta 5.5.1 numa maquina onde `apt install tesseract-ocr` so' trouxe
    4.1.1 -- os caminhos default de busca de cada versao nao batem). Procura
    nos locais usuais antes de desistir; devolve None se TESSDATA_PREFIX ja'
    estiver configurado (respeita o que foi setado explicitamente) ou se
    nenhum candidato tiver eng.traineddata."""
    if os.environ.get('TESSDATA_PREFIX'):
        return None
    for candidato in _TESSDATA_CANDIDATOS:
        if os.path.isfile(os.path.join(candidato, 'eng.traineddata')):
            return candidato
    return None


# PSMs candidatos pro slider de --calibrar -- os que fazem sentido pra um
# digito isolado (SINGLE_CHAR fica de fora do default por ser documentado
# como voltado pro motor legado, oem=0; ver comentario em LeitorDeNumero).
_PSM_CANDIDATOS = ()
if TESSEROCR_AVAILABLE:
    _PSM_CANDIDATOS = (
        ('SINGLE_BLOCK', PSM.SINGLE_BLOCK),
        ('SINGLE_LINE', PSM.SINGLE_LINE),
        ('SINGLE_WORD', PSM.SINGLE_WORD),
        ('SINGLE_CHAR', PSM.SINGLE_CHAR),
        ('RAW_LINE', PSM.RAW_LINE),
    )


@dataclass
class OcrParams:
    """Parametros do pre-processamento antes do OCR -- os "sliders de
    interesse" pedidos, tudo que ler() faz entre o recorte bruto e o texto
    que sai do Tesseract."""
    # Defaults calibrados ao vivo com --calibrar numa camera real (nao mais
    # os chutes iniciais) -- ver conversa/registro da calibracao.
    blur_kernel: int = 3       # tem que ser impar (corrigido na leitura)
    clahe_clip: float = 9.8
    clahe_tile: int = 2
    open_kernel: int = 4
    upscale: int = 4
    border: int = 4
    min_conf: float = 0.03
    psm_indice: int = 3        # indice em _PSM_CANDIDATOS -- 3 = SINGLE_CHAR
    # False = THRESH_BINARY normal (fundo claro, digito escuro -- o que o
    # LSTM do Tesseract espera). Toggle de seguranca caso a base de verdade
    # tenha a polaridade oposta -- teste ao vivo em vez de adivinhar.
    inverter: bool = False


# ── Formas: MESMOS filtros/limiares de RDPformas.py ─────────────────────────

def angulo_valido(approx, limite_min_graus=20, limite_max_graus=150):
    n = len(approx)
    if n < 3:
        return False

    for i in range(n):
        p1 = approx[i][0]
        p2 = approx[(i + 1) % n][0]
        p3 = approx[(i + 2) % n][0]

        u = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v = np.array([p3[0] - p2[0], p3[1] - p2[1]])

        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        if norm_u == 0 or norm_v == 0:
            continue

        cos_theta = np.clip(np.dot(u, v) / (norm_u * norm_v), -1.0, 1.0)
        angulo = math.degrees(math.acos(cos_theta))

        if angulo < limite_min_graus or angulo > limite_max_graus:
            return False

    return True


def detect_shape(contour):
    perim_filho = cv2.arcLength(contour, True)
    if perim_filho == 0:
        return "UNKNOWN"

    area_filho = cv2.contourArea(contour)
    circularidade_filho = (4 * math.pi * area_filho) / (perim_filho ** 2) if perim_filho > 0 else 0

    epsilon = 0.02 * perim_filho
    epsilon_coarse = 0.05 * perim_filho
    approx = cv2.approxPolyDP(contour, epsilon, True)
    approx_coarse = cv2.approxPolyDP(contour, epsilon_coarse, True)
    quantidade_de_pontos = len(approx)
    n_coarse = len(approx_coarse)

    hull = cv2.convexHull(contour)
    area_hull = cv2.contourArea(hull)
    perim_hull = cv2.arcLength(hull, True)
    circularidade_hull = (4 * math.pi * area_hull) / (perim_hull ** 2) if perim_hull > 0 else 0
    epsilon_hull = 0.02 * perim_hull if perim_hull > 0 else epsilon
    approx2 = cv2.approxPolyDP(hull, epsilon_hull, True)
    qtd_pontos_hull = len(approx2)

    if not angulo_valido(approx, 5, 175):
        return "UNKNOWN"
    if not angulo_valido(approx2, 20, 135):
        return "UNKNOWN"

    if quantidade_de_pontos == 4 or qtd_pontos_hull == 4:
        return "UNKNOWN"

    if (quantidade_de_pontos == 3 or n_coarse == 3) and (0.450 <= circularidade_filho <= 0.730):
        return "TRIANGULO"

    if quantidade_de_pontos == 6 and (0.700 <= circularidade_filho <= 0.970):
        return "HEXAGONO"

    if (4 <= qtd_pontos_hull <= 6) and (0.750 <= circularidade_hull <= 0.970) and quantidade_de_pontos >= 5:
        return "ESTRELA"

    return "UNKNOWN"


def _classify_digit_345(thresh):
    h, w = thresh.shape[:2]
    if h == 0 or w == 0:
        return None, 0.0

    cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or len(cnts) == 0:
        return None, 0.0

    holes = sum(1 for row in hierarchy[0] if row[3] != -1)
    if holes >= 1:
        conf = min(1.0, 0.55 + holes * 0.2)
        return '4', conf

    top = thresh[:max(1, h // 4), :]
    filled_cols = int(np.sum(np.any(top > 0, axis=0)))
    bar_ratio = filled_cols / max(w, 1)
    if bar_ratio > 0.5:
        conf = min(1.0, 0.45 + bar_ratio * 0.5)
        return '5', conf

    return '3', 0.40


class LeitorDeNumero:
    """Mesma logica de RDPvisao.read_number_in_contour, so' sem `self` de
    no ROS -- a engine tesserocr fica aqui em vez de num atributo de Node."""

    def __init__(self):
        self._ocr = None
        self._ocr_lock = threading.Lock()
        if TESSEROCR_AVAILABLE:
            print('tesserocr disponivel -- usando pra leitura de digito.')
            try:
                tessdata = _achar_tessdata()
                kwargs = {'path': tessdata} if tessdata else {}
                # PSM inicial -- sobrescrito por frame em ler() conforme
                # OcrParams.psm_indice, mas precisa de um valor de largada.
                # SINGLE_CHAR e' documentado pelo proprio Tesseract como
                # voltado pro motor LEGADO (oem=0); a suspeita inicial era
                # que isso desse resultado ruim com oem=3 (LSTM), mas
                # calibrando ao vivo (--calibrar) com a polaridade do
                # threshold ja corrigida, SINGLE_CHAR se saiu bem -- o
                # problema real era a polaridade, nao o PSM.
                self._ocr = PyTessBaseAPI(psm=PSM.SINGLE_CHAR, oem=3, **kwargs)
                # Whitelist com TODOS os digitos, nao so' '345' -- uma
                # whitelist estreita demais briga com o vocabulario que o
                # LSTM foi treinado a reconhecer e pode piorar a leitura em
                # vez de ajudar. So' aceitamos '345' como resultado VALIDO
                # depois, no filtro em ler() -- a engine pode "pensar" em
                # qualquer digito, nos e' que decidimos o que vale.
                self._ocr.SetVariable('tessedit_char_whitelist', '0123456789')
            except Exception as e:
                self._ocr = None
                print(f'aviso: falha ao iniciar tesserocr ({e}) -- seguindo sem ele',
                      file=sys.stderr)
        elif PYTESSERACT_AVAILABLE:
            print('pytesseract disponivel -- usando como leitor alternativo.')
        else:
            print('nem tesserocr nem pytesseract encontrados -- caindo pra '
                  'classificacao de digito por contorno.', file=sys.stderr)

        # Guarda o ultimo recorte pre-processado pra quem quiser MOSTRAR
        # exatamente o que foi mandado pro OCR (ver --debug-ocr em main()) --
        # sem isso, um erro de leitura pode ser do OCR ou do pre-
        # processamento, e nao da' pra saber qual sem inspecionar essa imagem.
        self.ultimo_thresh = None

    def close(self) -> None:
        if self._ocr is not None:
            try:
                self._ocr.End()
            except Exception:
                pass

    def ler(self, frame, contour, params: OcrParams = None):
        p = params or OcrParams()

        x, y, w, h = cv2.boundingRect(contour)
        margin = max(5, min(w, h) // 6)
        x1 = max(0, x + margin)
        y1 = max(0, y + margin)
        x2 = min(frame.shape[1], x + w - margin)
        y2 = min(frame.shape[0], y + h - margin)
        if x2 <= x1 or y2 <= y1:
            return None, 0.0, None

        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi

        # Blur leve (ruido do sensor) + CLAHE (contraste local) antes do
        # Otsu -- sem isso, luz desigual no recorte (sombra de um lado,
        # reflexo do outro) faz o threshold global cortar o digito pela
        # metade. Preprocessamento que faltava aqui (RDPformas.py so' tinha
        # o Otsu cru); e' o mesmo da versao standalone que funcionou melhor.
        k_blur = max(1, p.blur_kernel | 1)  # forca impar (GaussianBlur exige)
        gray = cv2.GaussianBlur(gray, (k_blur, k_blur), 0)
        clahe_tile = max(1, p.clahe_tile)
        gray = cv2.createCLAHE(clipLimit=max(0.1, p.clahe_clip),
                                tileGridSize=(clahe_tile, clahe_tile)).apply(gray)

        # SEM _INV por padrao: o LSTM do Tesseract e' treinado quase todo em
        # texto ESCURO sobre fundo CLARO (como uma pagina escaneada). Uma
        # imagem com a polaridade invertida (fundo escuro, digito claro)
        # pode parecer perfeitamente legivel pra um humano e ainda assim
        # confundir bastante o modelo -- e' exatamente esse footgun que
        # fazia a leitura sair sempre errada mesmo com "a imagem boa".
        # `inverter` fica de escape hatch, testavel ao vivo em --calibrar,
        # caso a base de verdade tenha a polaridade oposta.
        modo_threshold = cv2.THRESH_BINARY_INV if p.inverter else cv2.THRESH_BINARY
        _, thresh = cv2.threshold(gray, 0, 255, modo_threshold + cv2.THRESH_OTSU)
        cor_fundo = 0 if p.inverter else 255

        # Mascara pelo contorno na resolucao ORIGINAL do recorte -- a
        # ampliacao acontece so' depois, num passo so', em vez de escalar o
        # contorno junto (mais simples e o resultado e' o mesmo).
        contour_local = (contour.reshape(-1, 2) - np.array([x1, y1])
                         ).reshape(-1, 1, 2).astype(np.int32)
        shape_mask = np.zeros(thresh.shape[:2], dtype=np.uint8)
        cv2.drawContours(shape_mask, [contour_local], -1, 255, cv2.FILLED)
        # Fora do contorno vira a MESMA cor de fundo do threshold -- bitwise_and
        # sozinho forcaria 0 (preto) fora da mascara, o que reintroduziria a
        # polaridade errada bem na borda do digito quando nao invertido.
        thresh[shape_mask == 0] = cor_fundo

        # Abertura morfologica -- limpa pontinhos residuais do threshold
        # ANTES de ampliar (ampliar ruido so' deixa ele maior e mais dificil
        # de distinguir do digito de verdade). Kernel 0 = pula a abertura.
        if p.open_kernel > 0:
            open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                      (p.open_kernel, p.open_kernel))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_kernel)

        upscale = max(1, p.upscale)
        thresh = cv2.resize(thresh, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)

        # Borda na mesma cor de fundo (dinamica com `inverter`) -- evita que
        # o digito grude na borda da imagem, que o Tesseract confunde com um
        # traco cortado.
        if p.border > 0:
            thresh = cv2.copyMakeBorder(thresh, p.border, p.border, p.border, p.border,
                                         cv2.BORDER_CONSTANT, value=cor_fundo)

        self.ultimo_thresh = thresh

        if self._ocr is not None:
            try:
                nome_psm, valor_psm = _PSM_CANDIDATOS[p.psm_indice % len(_PSM_CANDIDATOS)]
                with self._ocr_lock:
                    self._ocr.SetPageSegMode(valor_psm)
                    self._ocr.SetImage(Image.fromarray(thresh))
                    # Explicito, igual o script de referencia que funciona --
                    # GetUTF8Text() e' documentado como "roda Recognize() se
                    # ainda nao rodou", mas nao custa nada tirar a duvida.
                    self._ocr.Recognize()
                    text = self._ocr.GetUTF8Text().strip()
                    ocr_conf = self._ocr.MeanTextConf() / 100.0
                # Print de diagnostico: sem isso, um resultado descartado
                # pelo filtro de confianca (ou vazio) cai pro fallback por
                # contorno em silencio, e o '3'/0.40 que sai la' e' IDENTICO
                # ao de um tesserocr que nunca leu nada -- impossivel
                # distinguir os dois so' olhando o resultado final.
                aceito = ocr_conf > p.min_conf and any(c in '345' for c in text)
                print(f'  [tesserocr psm={nome_psm}] leu {text!r} conf={ocr_conf:.2f} '
                      f'{"(aceito)" if aceito else "(descartado -- conf baixa ou digito fora de 345)"}',
                      file=sys.stderr)
                if aceito:
                    for c in text:
                        if c in '345':
                            return c, ocr_conf, 'tesserocr'
            except Exception as e:
                print(f'  [tesserocr] excecao: {e}', file=sys.stderr)

        if PYTESSERACT_AVAILABLE:
            try:
                text = pytesseract.image_to_string(
                    thresh, config='--psm 10 --oem 3 -c tessedit_char_whitelist=345')
                for c in text.strip():
                    if c in '345':
                        return c, 0.55, 'pytesseract'
            except Exception:
                pass

        # Chegou aqui: tesserocr indisponivel, sem confianca suficiente, ou
        # nao leu nenhum digito valido -- e pytesseract tambem nao ajudou.
        # O que sai daqui e' um PALPITE cru por forma de contorno, bem menos
        # confiavel que os dois de cima (ver fonte='heuristica' em quem
        # consome o resultado).
        digito, conf = _classify_digit_345(thresh)
        return digito, conf, 'heuristica'


def _abrir_camera(device: str, width: int, height: int):
    fonte = int(device) if device.isdigit() else device
    cap = cv2.VideoCapture(fonte, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f'nao foi possivel abrir a camera em {device!r}')
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def processar_frame(frame, aruco_detector, leitor, ocr_params: OcrParams = None):
    """Mesma logica do miolo de RDPvisao.process_frame -- ArUco + gabarito
    (contorno pai circular) + forma filha + leitura de numero -- sem as
    partes especificas de ROS/missao (mensagem, cache de alvo, fila
    assincrona). Devolve (frame_anotado, deteccoes) onde deteccoes e' uma
    lista de dicts {shape, numero, confianca, cx, cy}."""
    height, width = frame.shape[:2]
    outputRDP = frame.copy()
    deteccoes = []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = aruco_detector.detectMarkers(gray)
    if ids is not None and len(ids) > 0:
        aruco.drawDetectedMarkers(outputRDP, corners, ids)

    # ── Pre-processamento (identico a RDPformas.py) ─────────────────────
    gaussblur = cv2.GaussianBlur(gray, (11, 11), 0)
    grad_x = cv2.Scharr(gaussblur, cv2.CV_16S, 1, 0)
    grad_y = cv2.Scharr(gaussblur, cv2.CV_16S, 0, 1)
    threshold_alta, _ = cv2.threshold(gaussblur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold_baixa = threshold_alta * 0.9
    bordas = cv2.Canny(grad_x, grad_y, threshold_baixa, threshold_alta, L2gradient=True)
    kernel = np.ones((4, 4), np.uint8)
    bordasfinal = cv2.morphologyEx(bordas, cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(bordasfinal, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is not None:
        hierarchy = hierarchy[0]

        for i, cnt in enumerate(contours):
            if hierarchy[i][3] != -1:
                continue

            area = cv2.contourArea(cnt)
            perimetro = cv2.arcLength(cnt, True)
            if perimetro == 0:
                continue
            circularidade = (4 * math.pi * area) / (perimetro ** 2)

            if not (circularidade > 0.70 and area > 3700):
                continue

            for j, filho_cnt in enumerate(contours):
                if hierarchy[j][3] != i:
                    continue

                area_filho = cv2.contourArea(filho_cnt)
                if area_filho < 2300:
                    continue

                shape = detect_shape(filho_cnt)
                if shape == "UNKNOWN":
                    continue

                x, y, w, h = cv2.boundingRect(filho_cnt)
                cv2.putText(outputRDP, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2)
                cv2.drawContours(outputRDP, [filho_cnt], -1, (0, 255, 0), 2)

                M = cv2.moments(filho_cnt)
                if M['m00'] == 0:
                    continue
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                numero, conf, fonte = leitor.ler(frame, filho_cnt, ocr_params)
                label = f"{shape}_{numero}" if numero else shape

                cv2.circle(outputRDP, (cx, cy), 10, (255, 255, 0), -1)
                cv2.putText(outputRDP,
                            f"{label} ({conf:.2f} {fonte})" if conf else label,
                            (cx - 30, cy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                deteccoes.append({
                    'shape': shape, 'numero': numero, 'confianca': conf, 'fonte': fonte,
                    'cx': cx, 'cy': cy,
                    'cx_norm': cx / width, 'cy_norm': cy / height,
                })

    return outputRDP, deteccoes


_JANELA_CONTROLES = 'controles OCR'


def _make_trackbars(p: OcrParams) -> None:
    cv2.namedWindow(_JANELA_CONTROLES, cv2.WINDOW_NORMAL)
    w = _JANELA_CONTROLES
    noop = lambda _v: None  # noqa: E731

    cv2.imshow(w, np.zeros((10, 380, 3), dtype=np.uint8))
    cv2.resizeWindow(w, 380, 280)

    cv2.createTrackbar('blur_kernel-1', w, p.blur_kernel - 1, 14, noop)
    cv2.createTrackbar('clahe_clip_x10', w, int(p.clahe_clip * 10), 100, noop)
    cv2.createTrackbar('clahe_tile', w, p.clahe_tile, 16, noop)
    cv2.createTrackbar('open_kernel', w, p.open_kernel, 10, noop)
    cv2.createTrackbar('upscale', w, p.upscale, 6, noop)
    cv2.createTrackbar('border', w, p.border, 30, noop)
    cv2.createTrackbar('min_conf_%', w, int(p.min_conf * 100), 100, noop)
    cv2.createTrackbar('inverter', w, int(p.inverter), 1, noop)
    if _PSM_CANDIDATOS:
        cv2.createTrackbar('psm_indice', w, p.psm_indice, len(_PSM_CANDIDATOS) - 1, noop)


def _read_trackbars() -> OcrParams:
    def g(name: str) -> int:
        return cv2.getTrackbarPos(name, _JANELA_CONTROLES)

    return OcrParams(
        blur_kernel=g('blur_kernel-1') + 1,
        clahe_clip=g('clahe_clip_x10') / 10.0,
        clahe_tile=g('clahe_tile'),
        open_kernel=g('open_kernel'),
        upscale=g('upscale'),
        border=g('border'),
        min_conf=g('min_conf_%') / 100.0,
        inverter=bool(g('inverter')),
        psm_indice=g('psm_indice') if _PSM_CANDIDATOS else 0,
    )


def _print_params(p: OcrParams) -> None:
    nome_psm = _PSM_CANDIDATOS[p.psm_indice % len(_PSM_CANDIDATOS)][0] if _PSM_CANDIDATOS else 'n/a'
    print('\n# ---- parametros de OCR ajustados (cole em LeitorDeNumero) ----')
    print(f'    blur_kernel = {p.blur_kernel}')
    print(f'    clahe_clip = {p.clahe_clip:.1f}')
    print(f'    clahe_tile = {p.clahe_tile}')
    print(f'    open_kernel = {p.open_kernel}')
    print(f'    upscale = {p.upscale}')
    print(f'    border = {p.border}')
    print(f'    min_conf = {p.min_conf:.2f}')
    print(f'    inverter = {p.inverter}')
    print(f'    psm = {nome_psm}')
    print('# ----------------------------------------------------------------\n')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--device', default='0',
                     help='indice (0, 1, ...) ou caminho (/dev/video2) da camera USB')
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--sem-janela', action='store_true',
                     help='nao abre janelas do OpenCV, so imprime no terminal')
    ap.add_argument('--calibrar', action='store_true',
                     help='sliders ao vivo pro pre-processamento do OCR (ignora --sem-janela); '
                          'ao sair (q / Ctrl+C) imprime os parametros ajustados')
    args = ap.parse_args()

    if args.calibrar and args.sem_janela:
        ap.error('--calibrar precisa de janela, nao combina com --sem-janela')

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
    aruco_params = aruco.DetectorParameters()
    aruco_detector = aruco.ArucoDetector(aruco_dict, aruco_params)

    leitor = LeitorDeNumero()

    cap = _abrir_camera(args.device, args.width, args.height)
    print(f'Camera aberta em {args.device!r} '
          f'({int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x'
          f'{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}). Ctrl+C para sair.\n')

    import time
    inicio = time.monotonic()

    ocr_params = OcrParams()
    if args.calibrar:
        _make_trackbars(ocr_params)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print('falha ao ler frame da camera -- device desconectado?')
                break

            if args.calibrar:
                ocr_params = _read_trackbars()

            anotado, deteccoes = processar_frame(frame, aruco_detector, leitor, ocr_params)

            t = time.monotonic() - inicio
            if not deteccoes:
                print(f'[{t:9.3f}s] nenhuma forma')
            else:
                for d in deteccoes:
                    print(f'[{t:9.3f}s] {d["shape"]:<10} numero={d["numero"] or "?"} '
                          f'conf={d["confianca"]:.2f} fonte={d["fonte"]} '
                          f'px=({d["cx"]},{d["cy"]}) norm=({d["cx_norm"]:.3f},{d["cy_norm"]:.3f})')

            if not args.sem_janela:
                cv2.imshow('RDPformas (q para sair)', anotado)
                # Mostra exatamente o recorte que foi mandado pro OCR -- se
                # o digito sair ilegivel/cortado AQUI, o problema e' o
                # pre-processamento, nao o Tesseract em si.
                if leitor.ultimo_thresh is not None:
                    cv2.imshow('recorte pro OCR', leitor.ultimo_thresh)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        leitor.close()
        if args.calibrar:
            _print_params(ocr_params)


if __name__ == '__main__':
    main()
