import cv2 as cv
import numpy as np
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from custom_msgs.msg import BouncingDetection
from cv_bridge import CvBridge
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy
)
import concurrent.futures
import cv2.aruco as aruco

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

class RDPvisao(Node):
    def __init__(self):
        super().__init__('rdpvisao_node')

        self.declare_parameter('vertical_camera_topic', '/vertical_camera/compressed')
        camera_topic = self.get_parameter('vertical_camera_topic').get_parameter_value().string_value

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        # subscreve no tópico da câmera
        self.create_subscription(CompressedImage, camera_topic, self.image_callback, qos)

        self.bridge = CvBridge()

        # Publisher para as detecções
        self.publisher = self.create_publisher(BouncingDetection, 'bouncing_detection', 10)


        # publisher para debug de imagem
        self.debug_pub = self.create_publisher(CompressedImage, 'bouncing_detection_image/compressed', qos)

        # OCR setup
        self._ocr = None
        if EASYOCR_AVAILABLE:
            self.get_logger().info('EasyOCR available — using it for digit reading.')
            try:
                self._ocr = easyocr.Reader(['en'], gpu=True, verbose=False)
            except Exception:
                self._ocr = None
                self.get_logger().warn('EasyOCR initialization failed — continuing without it')
        elif PYTESSERACT_AVAILABLE:
            self.get_logger().info('pytesseract available — using it as fallback digit reader.')
        else:
            self.get_logger().warn('Neither EasyOCR nor pytesseract found. Falling back to contour-based digit classification')

        # Thread pool para OCR assíncrono
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self._pending_ocr = []  # list of tuples (future, header, shape, bcx, bcy)
        self._jpeg_quality = 60

        # ArUco detector
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
        self.aruco_params = aruco.DetectorParameters_create()

        # Latched target
        self._cached_target_calculated = False
        self._cached_target_base = ''

        # ArUco flicker guard: soft-match is suppressed while ArUco was recently
        # detected. Without this, frames where ids=None leave contains_aruco=False,
        # causing the gabarito hexagon to trigger a soft-match toward the ArUco.
        self._aruco_seen_frames = 0
        self._ARUCO_SEEN_PERSISTENCE = 8  # frames (~0.4s at 20Hz)

        # Ajustes para telemetria de debug
        self.declare_parameter('debug_pub_interval', 0.5)
        self._debug_pub_interval = float(self.get_parameter('debug_pub_interval').get_parameter_value().double_value)
        self._debug_last_pub_time = 0.0
        self._debug_max_width = 640


    def angulo_valido(self, approx, limite_min_graus=20, limite_max_graus=150):
        n = len(approx)
        if n < 3: return False
        
        for i in range(n):
            # Pega 3 pontos seguidos (fazendo o wrap com o resto da divisão para o último ponto)
            p1 = approx[i][0]
            p2 = approx[(i + 1) % n][0] # Vértice do ângulo
            p3 = approx[(i + 2) % n][0]
            
            # Cria os vetores u e v
            u = np.array([p1[0] - p2[0], p1[1] - p2[1]])
            v = np.array([p3[0] - p2[0], p3[1] - p2[1]])
            
            # Produto escalar e normas
            dot_product = np.dot(u, v)
            norm_u = np.linalg.norm(u)
            norm_v = np.linalg.norm(v)
            
            # Evita divisão por zero
            if norm_u == 0 or norm_v == 0: continue
            
            # Calcula o ângulo em graus
            cos_theta = dot_product / (norm_u * norm_v)
            # Clip para evitar erros de precisão numérica fora de [-1, 1]
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            angulo = math.degrees(math.acos(cos_theta))
            
            if angulo < limite_min_graus or angulo > limite_max_graus:
                return False # Se achar um ângulo "fechado" demais, descarta o contorno
                
        return True

    # Critério de divisibilidade
    def calculate_target_number(self, aruco_id):
        for n in [5, 4, 3]:
            if aruco_id % n == 0:
                return str(n)
        return None

    def _classify_digit_345(self, thresh):
        h, w = thresh.shape[:2]
        if h == 0 or w == 0:
            return None, 0.0

        cnts, hierarchy = cv.findContours(thresh.copy(), cv.RETR_CCOMP,
                                          cv.CHAIN_APPROX_SIMPLE)
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

    def read_number_in_contour(self, frame, contour):
        x, y, w, h = cv.boundingRect(contour)
        margin = max(5, min(w, h) // 6)
        x1 = max(0, x + margin)
        y1 = max(0, y + margin)
        x2 = min(frame.shape[1], x + w - margin)
        y2 = min(frame.shape[0], y + h - margin)
        if x2 <= x1 or y2 <= y1:
            return None, 0.0

        roi = frame[y1:y2, x1:x2]
        gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY) if roi.ndim == 3 else roi

        scale = max(1, 80 // min(gray.shape[:2]))
        if scale > 1:
            gray = cv.resize(gray, None, fx=scale, fy=scale,
                              interpolation=cv.INTER_CUBIC)

        _, thresh = cv.threshold(gray, 0, 255,
                                  cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

        contour_local = ((contour.reshape(-1, 2) - np.array([x1, y1])) * scale
                         ).reshape(-1, 1, 2).astype(np.int32)
        shape_mask = np.zeros(thresh.shape[:2], dtype=np.uint8)
        cv.drawContours(shape_mask, [contour_local], -1, 255, cv.FILLED)
        thresh = cv.bitwise_and(thresh, thresh, mask=shape_mask)

        # easyocr
        if self._ocr is not None:
            try:
                results = self._ocr.readtext(thresh, allowlist='345',
                                             detail=1, paragraph=False)
                for (_, text, ocr_conf) in results:
                    if ocr_conf > 0.4:
                        for c in text.strip():
                            if c in '345':
                                conf = 0.90 if ocr_conf > 0.7 else 0.65
                                return c, conf
            except Exception:
                pass

        # pytesseract
        if PYTESSERACT_AVAILABLE:
            try:
                text = pytesseract.image_to_string(
                    thresh,
                    config='--psm 10 --oem 3 -c tessedit_char_whitelist=345'
                )
                for c in text.strip():
                    if c in '345':
                        return c, 0.55
            except Exception:
                pass

        # contour based
        digit, conf = self._classify_digit_345(thresh)
        return digit, conf

    def detect_shape(self, contour):
        perim_filho = cv.arcLength(contour, True)
        if perim_filho == 0:
            return "UNKNOWN"

        area_filho = cv.contourArea(contour)
        circularidade_filho = (4 * math.pi * area_filho) / (perim_filho ** 2) if perim_filho > 0 else 0

        # Fine epsilon for hexagon; coarse epsilon for triangle robustness
        epsilon       = 0.02 * perim_filho
        epsilon_coarse = 0.05 * perim_filho
        approx        = cv.approxPolyDP(contour, epsilon, True)
        approx_coarse = cv.approxPolyDP(contour, epsilon_coarse, True)
        quantidade_de_pontos = len(approx)
        n_coarse             = len(approx_coarse)

        # Convex hull — use hull's OWN perimeter for epsilon so star hulls
        # (pentagon) are not over-smoothed by the star's much larger perim_filho.
        hull = cv.convexHull(contour)
        area_hull = cv.contourArea(hull)
        perim_hull = cv.arcLength(hull, True)
        circularidade_hull = (4 * math.pi * area_hull) / (perim_hull ** 2) if perim_hull > 0 else 0
        epsilon_hull = 0.02 * perim_hull if perim_hull > 0 else epsilon
        approx2      = cv.approxPolyDP(hull, epsilon_hull, True)
        qtd_pontos_hull = len(approx2)

        # Raw-contour angle check: upper limit loosened to 175 so that slightly
        # noisy star contours with near-collinear points are not rejected.
        if not self.angulo_valido(approx, 5, 175):
            return "UNKNOWN"
        if not self.angulo_valido(approx2, 20, 135):
            return "UNKNOWN"

        # Checagem das formas geométricas
        if quantidade_de_pontos == 4 or qtd_pontos_hull == 4:
            return "UNKNOWN"

        # Triangle: accept 3 pts with fine OR coarse epsilon to handle real-world
        # contours where slight noise gives 4 pts at fine epsilon.
        if (quantidade_de_pontos == 3 or n_coarse == 3) and (0.450 <= circularidade_filho <= 0.730):
            return "TRIANGULO"

        if quantidade_de_pontos == 6 and (0.700 <= circularidade_filho <= 0.970):
            return "HEXAGONO"

        # Star: hull must look like a pentagon (4-6 pts after proper epsilon).
        if (4 <= qtd_pontos_hull <= 6) and (0.750 <= circularidade_hull <= 0.970) and quantidade_de_pontos >= 5:
            return "ESTRELA"

        return "UNKNOWN"

    def _submit_async_ocr(self, frame, contour, header, shape, bcx, bcy, frame_w, frame_h):
        # task rodando no background para fazer a leitura do número dentro do contorno
        future = self._executor.submit(self.read_number_in_contour, frame, contour)
        # armazena o tamanho do frame para computar erros normalizados
        self._pending_ocr.append((future, header, shape, bcx, bcy, frame_w, frame_h))

    def _process_pending_ocr(self):
        remaining = []
        for item in self._pending_ocr:
            if len(item) == 5:
                future, header, shape, bcx, bcy = item
                frame_w = None
                frame_h = None
            else:
                future, header, shape, bcx, bcy, frame_w, frame_h = item
            if future.done():
                try:
                    number, conf = future.result(timeout=0)
                except Exception:
                    number, conf = None, 0.0

                det_msg = BouncingDetection()
                det_msg.header = header

                visible_bases = []
                visible_bases_x = []
                visible_bases_y = []
                visible_bases_conf = []

                base_id = f"{shape}_{number}" if number else shape
                visible_bases.append(base_id)
                # computa os erros normalizados se o tamanho do frame foi informado
                if frame_w and frame_h:
                    bx_err = float((bcx - frame_w  / 2.0) / (frame_w  / 2.0))
                    by_err = float((bcy - frame_h / 2.0) / (frame_h / 2.0))
                else:
                    bx_err = 0.0
                    by_err = 0.0
                visible_bases_x.append(bx_err)
                visible_bases_y.append(by_err)
                visible_bases_conf.append(float(conf) if conf else 0.0)

                det_msg.visible_bases = visible_bases
                det_msg.visible_bases_x_error = visible_bases_x
                det_msg.visible_bases_y_error = visible_bases_y
                det_msg.visible_bases_confidence = visible_bases_conf

                if self._cached_target_calculated and det_msg.visible_bases:
                    det_msg.target_calculated = True
                    det_msg.target_base = self._cached_target_base
                    if base_id == self._cached_target_base:
                        det_msg.target_base_in_sight = True
                try:
                    self.publisher.publish(det_msg)
                    self.get_logger().info("ACHEEEEEEI!!!!")
                except Exception:
                    pass
            else:
                remaining.append((future, header, shape, bcx, bcy))
        self._pending_ocr = remaining

    def image_callback(self, msg):
        try:
            self._process_pending_ocr()
        except Exception:
            pass

        # Converte a imagem
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return

        height, width = frame.shape[:2]
        outputRDP = frame.copy()

        det_msg = BouncingDetection()
        det_msg.header = msg.header

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Detecçao do ArUco
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict,
                                              parameters=self.aruco_params)

        if ids is not None and len(ids) > 0:
            aruco.drawDetectedMarkers(outputRDP, corners, ids)

            c  = corners[0][0]
            cx = (c[0][0] + c[2][0]) / 2.0
            cy = (c[0][1] + c[2][1]) / 2.0

            det_msg.aruco_detected = True
            det_msg.aruco_id       = int(ids[0][0])
            det_msg.aruco_x_error  = float((cx - width  / 2.0) / (width  / 2.0))
            det_msg.aruco_y_error  = float((cy - height / 2.0) / (height / 2.0))

            # Forma ao redor do ArUco
            blurred  = cv.GaussianBlur(gray, (5, 5), 0)
            edges    = cv.Canny(blurred, 50, 150)
            aruco_contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL,
                                               cv.CHAIN_APPROX_SIMPLE)

            aruco_shape = "UNKNOWN"
            for cnt in sorted(aruco_contours, key=cv.contourArea, reverse=True):
                if cv.contourArea(cnt) < 500:
                    continue
                shape = self.detect_shape(cnt)
                if shape == "UNKNOWN":
                    continue
                M = cv.moments(cnt)
                if M['m00'] == 0:
                    continue
                scx = int(M['m10'] / M['m00'])
                scy = int(M['m01'] / M['m00'])
                if abs(scx - cx) < 150 and abs(scy - cy) < 150:
                    aruco_shape = shape
                    cv.putText(outputRDP, f"GABARITO: {shape}",
                                (scx, scy - 10), cv.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2)
                    break

            det_msg.aruco_shape = aruco_shape

            # Divisibilidade
            if aruco_shape != "UNKNOWN":
                target_number = self.calculate_target_number(det_msg.aruco_id)
                if target_number is not None:
                    det_msg.target_calculated = True
                    det_msg.target_base = f"{aruco_shape}_{target_number}"
                    cv.putText(outputRDP, f"TARGET: {det_msg.target_base}",
                                (20, 40), cv.FONT_HERSHEY_SIMPLEX,
                                1.0, (255, 0, 0), 2)

        # Latch the target
        if det_msg.target_calculated:
            self._cached_target_calculated = True
            self._cached_target_base = det_msg.target_base
        elif self._cached_target_calculated:
            det_msg.target_calculated = True
            det_msg.target_base = self._cached_target_base

        # Update ArUco flicker counter: keep it high while ArUco is present,
        # decay after it leaves the frame.
        if det_msg.aruco_detected:
            self._aruco_seen_frames = self._ARUCO_SEEN_PERSISTENCE
        else:
            self._aruco_seen_frames = max(0, self._aruco_seen_frames - 1)

        # ==========================================================
        # 1. PRÉ-PROCESSAMENTO
        # ==========================================================
        img_cinza = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # blur Gaussiano pesado (11x11) para matar o ruído
        gaussblur = cv.GaussianBlur(img_cinza, (11, 11), 0)

        # filtro Scharr (mais sensível e preciso que o Sobel)
        grad_x = cv.Scharr(gaussblur, cv.CV_16S, 1, 0)
        grad_y = cv.Scharr(gaussblur, cv.CV_16S, 0, 1)


        # Otsu para achar o threshold ideal baseado na luz atual
        threshold_alta, _ = cv.threshold(gaussblur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            
        threshold_baixa = threshold_alta * 0.9
            
        # Canny recebendo os gradientes e calculando com L2gradient
        bordas = cv.Canny(grad_x, grad_y, threshold_baixa, threshold_alta, L2gradient=True)

        """
        binaria = cv.adaptiveThreshold(gaussblur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
        bordas = cv.Canny(binaria, 50, 150)
        """

        # cola as quinas e linhas que ficaram falhadas/pontilhadas
        kernel = np.ones((4,4), np.uint8)
        bordasfinal = cv.morphologyEx(bordas, cv.MORPH_CLOSE, kernel)


        # ==========================================================
        # 2. ENCONTRANDO CONTORNOS E FILTRANDO O LIXO
        # ==========================================================
        contours, hierarchy = cv.findContours(bordasfinal, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

        outputRDP = frame.copy()

        visible_bases      = []
        visible_bases_x    = []
        visible_bases_y    = []
        visible_bases_conf = []

        if hierarchy is not None:
            hierarchy = hierarchy[0]

            for i, cnt in enumerate(contours):
                paiIndex = hierarchy[i][3]

                if paiIndex == -1:
                    area = cv.contourArea(cnt)
                    perimetro = cv.arcLength(cnt, True)
                    if perimetro == 0: continue

                    circularidade = (4 * math.pi * area) / (perimetro ** 2)

                    if circularidade > 0.70 and area > 3700:
                        for j, filho_cnt in enumerate(contours):
                            if hierarchy[j][3] == i:
                                area_filho = cv.contourArea(filho_cnt)
                                if area_filho < 2300: continue
                                perim_filho = cv.arcLength(filho_cnt, True)
                                if perim_filho == 0: continue
                                circularidade_filho = (4 * math.pi * area_filho) / (perim_filho ** 2)

                                # Fine + coarse epsilon: coarse is used as a fallback
                                # for triangles whose noisy contour approximates to 4 pts.
                                epsilon        = 0.02 * perim_filho
                                epsilon_coarse = 0.05 * perim_filho
                                approx         = cv.approxPolyDP(filho_cnt, epsilon, True)
                                approx_coarse  = cv.approxPolyDP(filho_cnt, epsilon_coarse, True)
                                quantidade_de_pontos = len(approx)
                                n_coarse             = len(approx_coarse)

                                hull = cv.convexHull(filho_cnt)
                                area_hull = cv.contourArea(hull)
                                perim_hull = cv.arcLength(hull, True)
                                circularidade_hull = (4 * math.pi * area_hull) / (perim_hull ** 2)

                                solidez = area_filho / float(area_hull) if area_hull > 0 else 0

                                # Use hull's OWN perimeter for epsilon — perim_filho of a
                                # star is much larger than perim_hull, causing over-smoothing.
                                epsilon_hull = 0.02 * perim_hull if perim_hull > 0 else epsilon
                                approx2      = cv.approxPolyDP(hull, epsilon_hull, True)
                                qtd_pontos_hull = len(approx2)

                                # Upper limit 175 (was 165) to tolerate near-collinear noise
                                # points in star contours without rejecting valid shapes.
                                if not self.angulo_valido(approx, 5, 175): continue
                                if not self.angulo_valido(approx2, 20, 135): continue

                                # Checagem das formas geométricas

                                if quantidade_de_pontos == 4 or qtd_pontos_hull == 4:
                                    continue

                                shape = "UNKNOWN"
                                if (quantidade_de_pontos == 3 or n_coarse == 3) and (0.450 <= circularidade_filho <= 0.730):
                                    shape = "TRIANGULO"
                                    x, y, w, h = cv.boundingRect(filho_cnt)
                                    cv.putText(outputRDP, "Triangulo", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                                    self.get_logger().info('[TRIANGULO] detectado')

                                elif quantidade_de_pontos == 6 and (0.700 <= circularidade_filho <= 0.970):
                                    shape = "HEXAGONO"
                                    x, y, w, h = cv.boundingRect(filho_cnt)
                                    cv.putText(outputRDP, "Hexagono", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                                    self.get_logger().info('[HEXAGONO] detectado')

                                elif (4 <= qtd_pontos_hull <= 6) and (0.750 <= circularidade_hull <= 0.970) and quantidade_de_pontos >= 5:
                                    shape = "ESTRELA"
                                    x, y, w, h = cv.boundingRect(filho_cnt)
                                    cv.putText(outputRDP, "Estrela", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                                    self.get_logger().info('[ESTRELA] detectada')

                                # Desenha os contornos aprovados em verde e os vértices em vermelho
                                cv.drawContours(outputRDP, [approx], -1, (0, 255, 0), 2)
                                for ponto in approx:
                                    px, py = ponto[0]
                                    cv.circle(outputRDP, (px, py), 4, (0, 0, 255), -1)

                                # Centroides
                                Mloc = cv.moments(filho_cnt)
                                if Mloc['m00'] == 0:
                                    bcx = 0
                                    bcy = 0
                                else:
                                    bcx = int(Mloc['m10'] / Mloc['m00'])
                                    bcy = int(Mloc['m01'] / Mloc['m00'])
                                bx_err = float((bcx - width  / 2.0) / (width  / 2.0))
                                by_err = float((bcy - height / 2.0) / (height / 2.0))

                                
                                # If any detected ArUco marker has one of its corners inside this child
                                # contour, treat this gabarito as containing an ArUco. In that case
                                # the numeric digit is occluded by the marker and we should NOT
                                # consider this a numbered base (avoid soft-match by shape-only).
                                contains_aruco = False
                                if ids is not None and len(corners) > 0:
                                    try:
                                        for ac in corners:
                                            ac_pts = ac[0]
                                            for pt in ac_pts:
                                                if cv.pointPolygonTest(filho_cnt, (float(pt[0]), float(pt[1])), False) >= 0:
                                                    contains_aruco = True
                                                    break
                                            if contains_aruco:
                                                break
                                    except Exception:
                                        contains_aruco = False

                                if contains_aruco:
                                    # log and skip numeric-read / soft-match for this contour
                                    cv.putText(outputRDP, "ARUCO_IN_GABARITO", (bcx, bcy - 25), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                    self.get_logger().info('[SKIP] Contour contains ArUco — treating as ArUco base; skipping numeric read/soft-match')
                                    # Do not add to visible_bases as a numbered base; rely on ArUco detection instead
                                    continue

                                # Read the number + detection confidence (sync or async)
                                if self._ocr is not None or PYTESSERACT_AVAILABLE:
                                    # submit async OCR (we'll publish a later update when it completes)
                                    try:
                                        self._submit_async_ocr(frame.copy(), filho_cnt.copy(), msg.header, shape, bcx, bcy, width, height)
                                    except Exception:
                                        pass
                                    number, num_conf = None, 0.0
                                else:
                                    number, num_conf = self.read_number_in_contour(frame, filho_cnt)

                                base_id = f"{shape}_{number}" if number else shape

                                visible_bases.append(base_id)
                                visible_bases_x.append(bx_err)
                                visible_bases_y.append(by_err)
                                visible_bases_conf.append(float(num_conf) if num_conf else 0.0)

                                cv.circle(outputRDP, (bcx, bcy), 10, (255, 255, 0), -1)
                                cv.putText(outputRDP,
                                            f"BASE {base_id} ({num_conf:.2f})" if num_conf else f"BASE {base_id}",
                                            (bcx - 30, bcy - 15), cv.FONT_HERSHEY_SIMPLEX,
                                            0.5, (255, 255, 0), 2)

                                # Match against the target (exact match, then soft match by shape)
                                if det_msg.target_calculated and base_id == det_msg.target_base:
                                    det_msg.target_base_in_sight = True
                                    det_msg.target_base_x_error  = bx_err
                                    det_msg.target_base_y_error  = by_err
                                    cv.circle(outputRDP, (bcx, bcy), 20, (0, 0, 255), 3)
                                elif (det_msg.target_calculated
                                      and not det_msg.target_base_in_sight
                                      and number is None
                                      and shape == det_msg.target_base.split('_')[0]
                                      and self._aruco_seen_frames == 0):
                                    # Soft match: shape matches target but number unreadable.
                                    # Guard: only fire when ArUco has been absent for
                                    # _ARUCO_SEEN_PERSISTENCE frames, preventing the gabarito
                                    # hexagon (around the ArUco) from being mistaken for the
                                    # target base during ArUco flickering.
                                    same_shape_readable = [b for b in visible_bases
                                                           if b.startswith(shape + '_')]
                                    if not same_shape_readable:
                                        self.get_logger().warn(
                                            f'Soft match: {shape} seen but number unreadable '
                                            f'(target={det_msg.target_base}) — using position anyway')
                                        det_msg.target_base_in_sight = True
                                        det_msg.target_base_x_error  = bx_err
                                        det_msg.target_base_y_error  = by_err
                                        cv.circle(outputRDP, (bcx, bcy), 20, (0, 165, 255), 3)

        det_msg.visible_bases            = visible_bases
        det_msg.visible_bases_x_error    = visible_bases_x
        det_msg.visible_bases_y_error    = visible_bases_y
        det_msg.visible_bases_confidence = visible_bases_conf

        try:
            self.publisher.publish(det_msg)
        except Exception as e:
            self.get_logger().error(f'Failed to publish detection: {e}')

        # publicando a imagem de debug
        try:
            debug_msg = self.bridge.cv2_to_compressed_imgmsg(outputRDP)
            debug_msg.header = msg.header
            self.debug_pub.publish(debug_msg)
            now_s = self.get_clock().now().nanoseconds * 1e-9
            last_s = getattr(self, '_debug_last_pub_time', 0.0)
            interval = float(self._debug_pub_interval)
            if (now_s - last_s) >= interval:
                h, w = outputRDP.shape[:2]
                if w > self._debug_max_width:
                   scale = float(self._debug_max_width) / float(w)
                   output_small = cv.resize(outputRDP, (int(w*scale), int(h*scale)),interpolation=cv.INTER_AREA)
                else:
                   output_small = outputRDP
                ret, enc = cv.imencode('.jpg', output_small,
                                        [int(cv.IMWRITE_JPEG_QUALITY), int(self._jpeg_quality)])
                if not ret:
                    self.get_logger().error('JPEG encoding failed for debug image')
                else:
                    debug_msg = CompressedImage()
                    debug_msg.format = 'jpeg'
                    debug_msg.data = np.array(enc).tobytes()
                    debug_msg.header = msg.header
                    self.debug_pub.publish(debug_msg)
                    self._debug_last_pub_time = now_s

        except Exception as e:
            self.get_logger().error(f'Failed to publish debug image: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = RDPvisao()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()
