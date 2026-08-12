#!/usr/bin/env python3
"""
Detector de bases de pouso por combinação de azul e amarelo.

A base de pouso é um quadrado com regiões azuis e amarelas adjacentes. Detectar
"azul" ou "amarelo" isoladamente enche a imagem de falsos positivos — céu,
reflexos, faixas do chão. O que identifica a base é a ADJACÊNCIA das duas cores,
e é isso que este detector procura: dilata as duas máscaras até que regiões
vizinhas se toquem, e fica com a interseção.

Portado de `cbr_2025/cv_utils/target_detector/base_detector.py`. O que mudou:

- Herda de `cv_nodes.detector.Detector` em vez do `TargetDetector` de 2025.
  A classe base de 2026 já entrega o que aquela fazia à mão: assinatura de
  `CompressedImage` com QoS de sensor, `CvBridge`, throttle por frequência e
  publicação de imagens de debug com resize e qualidade configuráveis.

- Publica em `/base_detector/detections`, seguindo a regra de nomenclatura do
  ARCHITECTURE.md (`/<nome_do_detector>/...`), e não no
  `/vertical_camera/classification` de 2025.

O que NÃO mudou, de propósito: a segmentação HSV, os limiares e a combinação por
dilatação. Esses números foram sintonizados na arena da CBR e são a parte do
código de 2025 que mais custou a acertar.

CONTRATO DE SAÍDA — a `vision_geometry` depende dele:

    vision_msgs/Detection2DArray em /base_detector/detections
      bbox.center.position.x/y  centro NORMALIZADO em [0,1]
      bbox.size_x/size_y        tamanho NORMALIZADO (x por largura, y por altura)
      bbox.center.theta         rotação da caixa, em RADIANOS
      results[0].hypothesis.class_id = 'landing_pad'

A normalização de `size_y` é pela ALTURA da imagem e a de `size_x` pela LARGURA,
inclusive para caixas rotacionadas. Isso é uma convenção, não uma medida
geométrica — mas é lossless desde que quem consome desfaça do mesmo jeito, que é
o que `GroundProjector::sizeInPixels` faz.
"""

import cv2
from detector.detector import Detector
import numpy as np
import rclpy
from sensor_msgs.msg import CompressedImage
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose


class BaseDetector(Detector):
    """Encontra bases de pouso por regiões que contêm azul E amarelo."""

    def __init__(self):
        super().__init__('base_detector')

        # ── Faixas HSV ────────────────────────────────────────────────────
        # Herdadas de 2025 sem alteração: sintonizadas na arena.
        self.declare_parameter('blue_lower', [100, 80, 50])
        self.declare_parameter('blue_upper', [130, 255, 255])
        self.declare_parameter('yellow_lower', [10, 100, 100])
        self.declare_parameter('yellow_upper', [40, 255, 255])

        # Morfologia por cor. O azul recebe kernel maior porque a região azul da
        # base é a externa e sofre mais com sombra nas bordas.
        self.declare_parameter('blue_kernel_size', 5)
        self.declare_parameter('blue_iterations', 3)
        self.declare_parameter('yellow_kernel_size', 3)
        self.declare_parameter('yellow_iterations', 2)

        # ── Combinação e filtros ──────────────────────────────────────────
        self.declare_parameter('combination_kernel_size', 15)
        self.declare_parameter('combination_iterations', 2)
        self.declare_parameter('combined_min_area', 0.001)   # fração da imagem
        self.declare_parameter('combined_max_area', 1.0)
        self.declare_parameter('combined_aspect_ratio_max', 2.0)
        self.declare_parameter('min_pixels_per_color', 100)
        self.declare_parameter('nms_center_distance_threshold', 0.1)

        self.declare_parameter('detection_topic', '/base_detector/detections')

        self.blue_lower = self._hsv_param('blue_lower')
        self.blue_upper = self._hsv_param('blue_upper')
        self.yellow_lower = self._hsv_param('yellow_lower')
        self.yellow_upper = self._hsv_param('yellow_upper')

        self.blue_kernel_size = int(self.get_parameter('blue_kernel_size').value)
        self.blue_iterations = int(self.get_parameter('blue_iterations').value)
        self.yellow_kernel_size = int(self.get_parameter('yellow_kernel_size').value)
        self.yellow_iterations = int(self.get_parameter('yellow_iterations').value)

        self.combination_kernel_size = int(self.get_parameter('combination_kernel_size').value)
        self.combination_iterations = int(self.get_parameter('combination_iterations').value)
        self.combined_min_area = float(self.get_parameter('combined_min_area').value)
        self.combined_max_area = float(self.get_parameter('combined_max_area').value)
        self.aspect_ratio_max = float(self.get_parameter('combined_aspect_ratio_max').value)
        self.min_pixels_per_color = int(self.get_parameter('min_pixels_per_color').value)
        self.nms_threshold = float(self.get_parameter('nms_center_distance_threshold').value)

        detection_topic = self.get_parameter('detection_topic').value
        self.detection_pub = self.create_publisher(Detection2DArray, detection_topic, 10)
        self.bbox_debug_pub = self.create_publisher(
            CompressedImage, '/base_detector/debug/bbox/compressed', 1)
        self.mask_debug_pub = self.create_publisher(
            CompressedImage, '/base_detector/debug/mask/compressed', 1)

        self.get_logger().info(f'base_detector publicando em {detection_topic}')

    def _hsv_param(self, name: str) -> np.ndarray:
        """Lê um parâmetro HSV de três inteiros como array do OpenCV."""
        value = self.get_parameter(name).value
        return np.array([int(v) for v in value], dtype=np.uint8)

    # ── Interface do Detector ─────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray, header) -> None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        blue_mask = self._segment(hsv, self.blue_lower, self.blue_upper,
                                  self.blue_kernel_size, self.blue_iterations)
        yellow_mask = self._segment(hsv, self.yellow_lower, self.yellow_upper,
                                    self.yellow_kernel_size, self.yellow_iterations)

        detections = self._find_combined_regions(blue_mask, yellow_mask)
        detections = self._filter_concentric(detections, frame.shape)

        self.detection_pub.publish(self._to_msg(detections, frame.shape, header))

        self._publish_debug(frame, blue_mask, yellow_mask, detections, header)

    # ── Visão ─────────────────────────────────────────────────────────────

    def _segment(self, hsv, lower, upper, kernel_size, iterations) -> np.ndarray:
        """
        Máscara de uma cor, com fechamento seguido de abertura.

        A ordem importa: CLOSE primeiro tapa os buracos internos da região
        (causados por reflexo e compressão JPEG); OPEN depois remove os pontos
        isolados. Invertido, o OPEN apagaria as regiões pequenas antes de o
        CLOSE ter chance de uni-las.
        """
        mask = cv2.inRange(hsv, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask

    def _find_combined_regions(self, blue_mask, yellow_mask) -> list:
        """Regiões que contêm as duas cores adjacentes."""
        img_height, img_width = blue_mask.shape[:2]
        img_area = img_height * img_width
        min_area_px = self.combined_min_area * img_area
        max_area_px = self.combined_max_area * img_area

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.combination_kernel_size, self.combination_kernel_size))

        # Dilatar as duas máscaras e interseccionar: onde as duas dilatações se
        # encontram, as regiões originais estavam a menos de ~2×kernel uma da
        # outra. É assim que "adjacente" vira uma operação.
        blue_dilated = cv2.dilate(blue_mask, kernel, iterations=self.combination_iterations)
        yellow_dilated = cv2.dilate(yellow_mask, kernel, iterations=self.combination_iterations)
        combined = cv2.bitwise_and(blue_dilated, yellow_dilated)

        # Somar a sobreposição direta cobre o caso em que as cores já se tocam,
        # que a dilatação sozinha poderia estreitar demais.
        combined = cv2.bitwise_or(combined, cv2.bitwise_and(blue_mask, yellow_mask))

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area_px or area > max_area_px:
                continue

            (cx, cy), (rect_w, rect_h), angle_deg = cv2.minAreaRect(contour)
            if rect_w <= 0 or rect_h <= 0:
                continue

            # A razão é sempre maior/menor, logo >= 1. O limite inferior de 0.5
            # que 2025 declarava era inalcançável e portanto morto; ficou só o
            # superior, que é o que de fato filtra.
            aspect_ratio = max(rect_w, rect_h) / min(rect_w, rect_h)
            if aspect_ratio > self.aspect_ratio_max:
                continue

            # Confirmar nas máscaras ORIGINAIS que as duas cores estão mesmo
            # ali. Sem isso, duas regiões distantes de cores erradas poderiam
            # se encontrar na dilatação e virar uma base fantasma.
            x, y, w, h = cv2.boundingRect(contour)
            blue_pixels = int(np.count_nonzero(blue_mask[y:y + h, x:x + w]))
            yellow_pixels = int(np.count_nonzero(yellow_mask[y:y + h, x:x + w]))
            if (blue_pixels < self.min_pixels_per_color or
                    yellow_pixels < self.min_pixels_per_color):
                continue

            detections.append({
                'bbox': (x, y, w, h),
                'center': (float(cx), float(cy)),
                'size': (float(rect_w), float(rect_h)),
                'angle': float(np.deg2rad(angle_deg)),
                'area': float(area),
                'aspect_ratio': float(aspect_ratio),
            })

        return detections

    def _filter_concentric(self, detections: list, image_shape) -> list:
        """
        Descarta detecções concêntricas, ficando com a de maior área.

        A base tem anéis de cor, então o mesmo alvo costuma gerar dois ou três
        contornos aninhados. Sem isso a missão trataria um único pouso como
        várias bases distintas.
        """
        if len(detections) <= 1:
            return detections

        img_height, img_width = image_shape[:2]
        threshold_px = self.nms_threshold * float(np.hypot(img_width, img_height))

        kept = []
        for det in sorted(detections, key=lambda d: d['area'], reverse=True):
            cx, cy = det['center']
            if any(np.hypot(cx - k['center'][0], cy - k['center'][1]) < threshold_px
                   for k in kept):
                continue
            kept.append(det)

        return kept

    # ── Publicação ────────────────────────────────────────────────────────

    def _to_msg(self, detections: list, image_shape, header) -> Detection2DArray:
        """Monta o Detection2DArray com a bbox normalizada."""
        msg = Detection2DArray()
        msg.header = header

        img_height, img_width = image_shape[:2]

        for det in detections:
            d = Detection2D()
            cx, cy = det['center']
            sz_w, sz_h = det['size']

            d.bbox.center.position.x = cx / img_width
            d.bbox.center.position.y = cy / img_height
            d.bbox.size_x = sz_w / img_width
            d.bbox.size_y = sz_h / img_height
            d.bbox.center.theta = det['angle']

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = 'landing_pad'
            hypothesis.hypothesis.score = 1.0
            d.results = [hypothesis]

            msg.detections.append(d)

        return msg

    def _publish_debug(self, frame, blue_mask, yellow_mask, detections, header) -> None:
        if not self._debug_should_publish():
            return

        if self.get_parameter('debug_image').value:
            bbox_img = frame.copy()
            for det in detections:
                cx, cy = det['center']
                sz_w, sz_h = det['size']
                rect = ((cx, cy), (sz_w, sz_h), np.rad2deg(det['angle']))
                cv2.drawContours(bbox_img, [cv2.boxPoints(rect).astype(int)], 0,
                                 (255, 0, 255), 2)
                x, y = det['bbox'][0], det['bbox'][1]
                cv2.putText(bbox_img, f"AR: {det['aspect_ratio']:.2f}", (x, max(y - 10, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            cv2.putText(bbox_img, f'Bases: {len(detections)}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            self._pub_debug(self.bbox_debug_pub, bbox_img, header)

        if self.get_parameter('debug_mask').value:
            overlay = frame.copy()
            overlay[blue_mask > 0] = (255, 0, 0)
            overlay[yellow_mask > 0] = (0, 255, 255)
            mask_img = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
            self._pub_debug(self.mask_debug_pub, mask_img, header)


def main(args=None):
    rclpy.init(args=args)
    node = BaseDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
