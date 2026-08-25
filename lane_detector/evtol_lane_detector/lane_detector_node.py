"""
lane_detector_node.py

Segue a linha azul (mission_line) na camera vertical usando visao classica:
segmentacao HSV -> N regioes de interesse horizontais -> centro de massa do
azul em cada regiao -> regressao (reta, grau configuravel) dos centroides ->
suavizacao temporal.

O ANGULO (theta) usa uma media ponderada pela distancia a mediana da janela
recente (ver _weighted_theta_smooth), nao uma EMA simples: uma leitura que
salta longe do resto da janela pesa pouco no resultado, o que evita tremor de
yaw quando uma deteccao ruidosa aparece isolada. O offset lateral
(x_centroid) continua com EMA simples -- so' o angulo tremia.

A deteccao da base circular vem de um no' separado, o circle_detector
(custom_msgs/msg/BaseCircle, /base_circle) -- este no' so' escuta a ultima
deteccao publicada la' pra recortar o disco da propria mascara antes da
regressao, evitando que a base contamine o fit da linha.

Publica custom_msgs/msg/LaneDirection em /lane_detection:
    theta       inclinacao (rad) da reta regredida em relacao a vertical da imagem;
                0 = linha perfeitamente vertical (alinhada com a direcao de voo)
    x_centroid  offset lateral (px) da reta na base da imagem em relacao ao centro
    y_centroid  y medio (px, relativo ao centro) dos centroides usados no frame
    area        numero de pixels azuis na mascara (proxy de confianca/tamanho da deteccao)
    lost        true quando a reta nao pode ser ajustada, mesmo apos o fallback de extrapolacao
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from custom_msgs.msg import BaseCircle, LaneDirection
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CompressedImage


_LANE_QOS = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
)

_DBG_QOS = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=5,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
)


@dataclass
class _FollowerState:
    """Estado do seguidor que precisa persistir entre frames sucessivos."""

    last_poly: Optional[np.ndarray] = None
    last_points: List[Tuple[int, int]] = field(default_factory=list)
    lost_counter: int = 0
    theta_window: List[float] = field(default_factory=list)
    smoothed_heading: Optional[float] = None
    ema_offset: Optional[float] = None


class LaneDetectorNode(Node):
    """Detecta e segue a linha azul (mission_line) na camera vertical."""

    def __init__(self):
        super().__init__('lane_detector')

        self.declare_parameter('image_topic', '/vertical_camera/compressed')
        self.declare_parameter('resize_width', 800)

        # Threshold HSV pra azul (ajuste pro ambiente/iluminacao)
        self.declare_parameter('blue_h_min', 100)
        self.declare_parameter('blue_h_max', 130)
        self.declare_parameter('blue_s_min', 80)
        self.declare_parameter('blue_s_max', 255)
        self.declare_parameter('blue_v_min', 40)
        self.declare_parameter('blue_v_max', 255)
        self.declare_parameter('morph_kernel_size', 3)

        # Quanto tempo (s) confiar na ultima deteccao de /base_circle antes de considerar
        # desatualizada e parar de recortar a base da mascara.
        self.declare_parameter('disc_info_max_age_s', 2.0)

        # Regioes de interesse horizontais
        self.declare_parameter('n_regions', 12)
        self.declare_parameter('region_thickness', 40)
        self.declare_parameter('min_pixels_per_region', 10)

        # Regressao dos centroides
        self.declare_parameter('poly_degree', 1)
        self.declare_parameter('min_points_for_fit', 3)

        # Suavizacao temporal + fallback quando a linha some
        self.declare_parameter('ema_alpha', 0.35)
        self.declare_parameter('max_lost_frames', 8)

        # Filtro de theta (angulo) por media ponderada pela distancia a mediana
        # da janela -- ver docstring do modulo. theta_smoothing_window quantas
        # leituras cruas entram na janela; theta_smoothing_sigma controla o quao
        # rapido o peso cai com a distancia (rad) -- menor sigma pune saltos com
        # mais forca, maior sigma aproxima de uma media simples da janela.
        self.declare_parameter('theta_smoothing_window', 5)
        self.declare_parameter('theta_smoothing_sigma', 0.3)

        self.declare_parameter('debug_mask', True)
        self.declare_parameter('debug_image', True)
        self.declare_parameter('debug_mask_overlay', True)

        self.publisher_ = self.create_publisher(LaneDirection, '/lane_detection', _LANE_QOS)
        self.debug_pub_ = self.create_publisher(CompressedImage, 'lane_detector/debug/compressed', _DBG_QOS)
        self.mask_pub_ = self.create_publisher(CompressedImage, 'lane_detector/mask/compressed', _DBG_QOS)
        # Mascara binaria + centroides das regioes + reta regredida, desenhados em cima da
        # mascara (em vez da imagem da camera) -- mais facil de ver se a regressao realmente
        # acompanha o blob azul.
        self.mask_overlay_pub_ = self.create_publisher(
            CompressedImage, 'lane_detector/mask_overlay/compressed', _DBG_QOS)

        image_topic = self.get_parameter('image_topic').value
        self.subscription = self.create_subscription(
            CompressedImage,
            image_topic,
            self.image_callback,
            _LANE_QOS,
        )

        # Deteccao da base circular vem de fora (ver docstring do modulo) -- so' guarda a
        # ultima leitura pra recortar a base da propria mascara.
        self.base_circle_sub_ = self.create_subscription(
            BaseCircle, '/base_circle', self._base_circle_callback, _DBG_QOS)
        self._last_disc: Optional[Tuple[float, float, float]] = None
        self._last_disc_time = None

        self.br = CvBridge()
        self._state = _FollowerState()
        self.get_logger().info(f'Lane detector (curve follower) started. Subscribed to {image_topic}')

    def _base_circle_callback(self, msg: BaseCircle):
        if msg.found:
            self._last_disc = (float(msg.x), float(msg.y), float(msg.radius))
            self._last_disc_time = self.get_clock().now()

    def _current_disc(self, max_age_s: float) -> Optional[Tuple[float, float, float]]:
        if self._last_disc is None or self._last_disc_time is None:
            return None
        age_s = (self.get_clock().now() - self._last_disc_time).nanoseconds / 1e9
        if age_s > max_age_s:
            return None
        return self._last_disc

    def _pub_debug(self, publisher, image, header):
        try:
            msg = self.br.cv2_to_compressed_imgmsg(image)
            msg.header = header
            publisher.publish(msg)
        except Exception as exc:
            self.get_logger().error(f'Failed to publish debug image: {exc}')

    def _build_blue_mask(self, frame_bgr):
        h_min = int(self.get_parameter('blue_h_min').value)
        h_max = int(self.get_parameter('blue_h_max').value)
        s_min = int(self.get_parameter('blue_s_min').value)
        s_max = int(self.get_parameter('blue_s_max').value)
        v_min = int(self.get_parameter('blue_v_min').value)
        v_max = int(self.get_parameter('blue_v_max').value)
        kernel_size = int(self.get_parameter('morph_kernel_size').value)

        blurred = cv2.GaussianBlur(frame_bgr, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        lower_blue = np.array([h_min, s_min, v_min], dtype=np.uint8)
        upper_blue = np.array([h_max, s_max, v_max], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # SEM erode/OPEN de proposito -- podem apagar uma linha fina quase por
        # completo antes da regiao de interesse contar pixel nenhum. CLOSE
        # fecha buracos sem afinar; dilate da' folga extra.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, None, iterations=1)
        return mask

    # -- regioes de interesse horizontais ---------------------------------

    @staticmethod
    def _region_centers(height: int, n_regions: int) -> List[int]:
        """Centros de y distribuidos uniformemente do topo (longe) a base (perto do drone)."""
        if n_regions <= 1:
            return [height // 2]
        step = (height - 1) / float(n_regions - 1)
        return [int(round(i * step)) for i in range(n_regions)]

    def _regions_centroids(
        self, mask: np.ndarray, n_regions: int, region_thickness: int, min_pixels_per_region: int
    ) -> List[Tuple[int, int]]:
        """Pra cada regiao horizontal (extremo esquerdo ao direito da imagem), acha o
        centro de massa do azul. Ordem topo -> base (ja' pronta pra regressao)."""
        height, width = mask.shape[:2]
        half_thickness = max(1, region_thickness // 2)
        points: List[Tuple[int, int]] = []

        for cy in self._region_centers(height, n_regions):
            y_low = max(0, cy - half_thickness)
            y_high = min(height, cy + half_thickness)

            band = mask[y_low:y_high, :]  # largura inteira: extremo esquerdo ao direito
            ys, xs = np.nonzero(band)

            if len(xs) >= min_pixels_per_region:
                cx = int(np.mean(xs))
                points.append((cx, cy))

        return points

    @staticmethod
    def _fit_curve(points: List[Tuple[int, int]], poly_degree: int, min_points_for_fit: int) -> Optional[np.ndarray]:
        if len(points) < min_points_for_fit:
            return None
        ys = np.array([p[1] for p in points], dtype=np.float64)
        xs = np.array([p[0] for p in points], dtype=np.float64)
        try:
            return np.polyfit(ys, xs, poly_degree)
        except np.linalg.LinAlgError:
            return None

    @staticmethod
    def _compute_outputs(coeffs: np.ndarray, width: int, height: int) -> Tuple[float, float]:
        """Retorna (heading_error, lateral_offset).

        heading_error: inclinacao (rad) da reta regredida em relacao a vertical da imagem
        (dx/dy avaliado na base -- pra poly_degree=1 e' a mesma inclinacao em qualquer y,
        entao "alinhar deixando a linha vertical" = trazer isso pra 0).
        """
        deriv = np.polyder(coeffs)

        y_base = height - 1
        x_base = float(np.polyval(coeffs, y_base))
        slope = float(np.polyval(deriv, y_base))  # dx/dy
        heading_error = math.atan2(slope, 1.0)
        lateral_offset = x_base - width / 2.0

        return heading_error, lateral_offset

    @staticmethod
    def _ema(prev: Optional[float], new: float, alpha: float) -> float:
        if prev is None:
            return new
        return alpha * prev + (1 - alpha) * new

    @staticmethod
    def _weighted_theta_smooth(window: List[float], new_theta: float, max_window: int, sigma: float) -> float:
        """Suaviza theta por media ponderada: mantem uma janela das ultimas
        leituras cruas (mutada in-place), e devolve a media dando menos peso
        as leituras mais distantes da MEDIANA da janela (nucleo gaussiano,
        peso 1.0 em d=0). Mediana como referencia, nao media ou leitura mais
        nova, porque e' insensivel a um unico salto isolado -- ao contrario
        da media (todos participam igual) ou de usar a propria leitura nova
        como referencia (peso maximo justo quando ela e' o salto).
        """
        window.append(new_theta)
        del window[:-max_window]

        if len(window) == 1:
            return new_theta

        reference = float(np.median(window))
        weights = [math.exp(-((t - reference) ** 2) / (2.0 * sigma * sigma)) for t in window]
        total_weight = sum(weights)
        if total_weight < 1e-9:
            return new_theta

        return sum(w * t for w, t in zip(weights, window)) / total_weight

    # -- pipeline completo -------------------------------------------------

    def image_callback(self, msg):
        try:
            # 'passthrough', nao 'bgr8': converter encoding aqui roteia pra
            # cv_bridge_boost (compilado p/ numpy 1.x) e segfaulta com numpy 2.x.
            # cv2.imdecode ja devolve BGR, entao passthrough da' o mesmo resultado.
            frame = self.br.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as exc:
            self.get_logger().error(f'Failed to convert image: {exc}')
            return

        resize_width = int(self.get_parameter('resize_width').value)
        scale = resize_width / float(frame.shape[1])
        frame = cv2.resize(frame, (resize_width, int(frame.shape[0] * scale)))
        height, width = frame.shape[:2]

        mask = self._build_blue_mask(frame)

        n_regions = int(self.get_parameter('n_regions').value)
        region_thickness = int(self.get_parameter('region_thickness').value)
        min_pixels_per_region = int(self.get_parameter('min_pixels_per_region').value)
        disc_info_max_age_s = float(self.get_parameter('disc_info_max_age_s').value)

        if bool(self.get_parameter('debug_mask').value):
            mask_dbg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            self._draw_regions(mask_dbg, height, width, n_regions, region_thickness)
            cv2.putText(mask_dbg, f'blue px={int(np.count_nonzero(mask))}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            self._pub_debug(self.mask_pub_, mask_dbg, msg.header)
        poly_degree = int(self.get_parameter('poly_degree').value)
        min_points_for_fit = int(self.get_parameter('min_points_for_fit').value)
        ema_alpha = float(self.get_parameter('ema_alpha').value)
        max_lost_frames = int(self.get_parameter('max_lost_frames').value)
        theta_smoothing_window = int(self.get_parameter('theta_smoothing_window').value)
        theta_smoothing_sigma = float(self.get_parameter('theta_smoothing_sigma').value)

        # Recorta a base da mascara ANTES de montar as regioes -- senao as
        # regioes proximas dela pegariam o centro do disco em vez da linha.
        disc = self._current_disc(disc_info_max_age_s)
        if disc is not None:
            regression_mask = mask.copy()
            dcx, dcy, dradius = disc
            cv2.circle(regression_mask, (int(round(dcx)), int(round(dcy))), int(round(dradius * 1.1)), 0, -1)
        else:
            regression_mask = mask

        points = self._regions_centroids(regression_mask, n_regions, region_thickness, min_pixels_per_region)
        coeffs = self._fit_curve(points, poly_degree, min_points_for_fit)

        if coeffs is None:
            # Throttled pra nao inundar o log a 20Hz -- serve pra diagnosticar se a
            # causa e' pouco ponto (regiao/limiar apertado demais pro traco fino)
            # ou outra coisa (ex: a linha realmente saiu do quadro).
            self.get_logger().warn(
                f'Nenhum ajuste possivel: {len(points)}/{n_regions} regioes com pixel '
                f'suficiente (min_pixels_per_region={min_pixels_per_region}), '
                f'blue px totais={int(np.count_nonzero(mask))}',
                throttle_duration_sec=1.0)

        state = self._state
        lane_msg = LaneDirection()

        used_points: List[Tuple[int, int]] = []
        used_poly = None
        status_text = ''

        if coeffs is not None:
            state.last_poly = coeffs
            state.last_points = points
            state.lost_counter = 0
            used_points = points
            used_poly = coeffs

            heading, offset = self._compute_outputs(coeffs, width, height)

            state.smoothed_heading = self._weighted_theta_smooth(
                state.theta_window, heading, theta_smoothing_window, theta_smoothing_sigma)
            state.ema_offset = self._ema(state.ema_offset, offset, ema_alpha)

            lane_msg.lost = False
            lane_msg.theta = float(state.smoothed_heading)
            lane_msg.x_centroid = int(round(state.ema_offset))
            status_text = f'heading={state.smoothed_heading:+.2f} offset={state.ema_offset:+.1f}px'
        else:
            state.lost_counter += 1
            if state.lost_counter <= max_lost_frames and state.last_poly is not None:
                # extrapola com o ultimo fit valido por alguns frames (dropout curto)
                used_poly = state.last_poly
                heading, offset = self._compute_outputs(state.last_poly, width, height)

                lane_msg.lost = False
                lane_msg.theta = float(state.smoothed_heading if state.smoothed_heading is not None else heading)
                lane_msg.x_centroid = int(round(state.ema_offset if state.ema_offset is not None else offset))
                status_text = f'EXTRAPOLATED (lost {state.lost_counter}/{max_lost_frames})'
            else:
                # perdido de verdade: zera estado
                lane_msg.lost = True
                lane_msg.theta = 0.0
                lane_msg.x_centroid = 0
                state.last_poly = None
                state.last_points = []
                # Zera a janela tambem, senao o filtro pondera a leitura nova
                # contra leituras de antes da linha sumir.
                state.theta_window = []
                state.smoothed_heading = None
                status_text = 'LOST'

        if used_points:
            mean_y = float(np.mean([p[1] for p in used_points]))
            lane_msg.y_centroid = int(round(mean_y - height / 2.0))
        else:
            lane_msg.y_centroid = 0

        lane_msg.area = int(np.count_nonzero(mask))

        self.publisher_.publish(lane_msg)

        if bool(self.get_parameter('debug_image').value):
            output = self._draw_debug(frame, used_points, used_poly, status_text, lane_msg.lost)
            self._pub_debug(self.debug_pub_, output, msg.header)

        if bool(self.get_parameter('debug_mask_overlay').value):
            mask_overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            self._draw_regions(mask_overlay, height, width, n_regions, region_thickness)
            if disc is not None:
                # circulo vermelho fino = exatamente o que foi recortado da mascara antes
                # da regressao (vindo do circle_detector) -- ajuda a conferir visualmente
                # se a base esta sendo bem separada da linha
                dcx, dcy, dradius = disc
                cv2.circle(mask_overlay, (int(round(dcx)), int(round(dcy))), int(round(dradius * 1.1)),
                           (0, 0, 255), 1)
            mask_overlay = self._draw_debug(mask_overlay, used_points, used_poly, status_text, lane_msg.lost)
            self._pub_debug(self.mask_overlay_pub_, mask_overlay, msg.header)

    # -- debug view ---------------------------------------------------------

    def _draw_regions(self, image: np.ndarray, height: int, width: int, n_regions: int, region_thickness: int) -> None:
        """Desenha o retangulo (extremo esquerdo ao direito, espessura region_thickness) de
        cada regiao de interesse, in-place -- deixa visivel no rqt onde cada ROI realmente
        fica e quao grossa ela e', pra ajudar a calibrar n_regions/region_thickness."""
        half_thickness = max(1, region_thickness // 2)
        for cy in self._region_centers(height, n_regions):
            y_low = max(0, cy - half_thickness)
            y_high = min(height - 1, cy + half_thickness)
            cv2.rectangle(image, (0, y_low), (width - 1, y_high), (255, 200, 0), 1)

    def _draw_debug(self, frame, points, poly_coeffs, status_text, lost) -> np.ndarray:
        out = frame.copy()
        height, width = out.shape[:2]

        for (x, y) in points:
            cv2.circle(out, (x, y), 4, (0, 255, 255), -1)

        if poly_coeffs is not None:
            ys = np.linspace(0, height - 1, 40)
            xs = np.polyval(poly_coeffs, ys)
            pts = np.stack([xs, ys], axis=1).astype(np.int32)
            cv2.polylines(out, [pts], isClosed=False, color=(0, 165, 255), thickness=2)

        cv2.line(out, (width // 2, 0), (width // 2, height), (255, 255, 255), 1)

        color = (0, 0, 255) if lost else (0, 255, 0)
        cv2.putText(out, status_text or 'LOST', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return out


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
