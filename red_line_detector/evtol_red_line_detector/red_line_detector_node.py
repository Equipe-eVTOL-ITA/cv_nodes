"""
red_line_detector_node.py

Detecta a mangueira vermelha entre os dois postes (fase4_itjbx) na camera
vertical. Ajusta uma reta a TODOS os pixels vermelhos via minimos quadrados
robusto (cv2.fitLine, sem assumir orientacao preferencial), em vez de varrer
por regioes verticais e ajustar y = f(x) -- essa abordagem antiga so'
funcionava com a mangueira ja perto de horizontal.

theta e' o angulo em relacao a VERTICAL (0 = mangueira vertical no quadro =
alinhado) -- mesma convencao do lane_detector, embora a mangueira apareça
naturalmente mais HORIZONTAL na aproximacao (e' perpendicular a rota).

Vermelho precisa de DUAS faixas de matiz (hue) no HSV -- a cor fica dividida
entre h≈0 e h≈180 (o matiz e' circular). Por isso ha' dois pares
(red_h1_min/max, red_h2_min/max) em vez de um so' como o azul do lane_detector.

Publica custom_msgs/msg/LaneDirection em /red_line_detection:
    theta       inclinacao (rad) em relacao a VERTICAL; 0 = alinhado
    x_centroid  posicao X media (px, relativo ao centro) dos pixels vermelhos
    y_centroid  offset (px) de onde a reta cruza a coluna central -- o que
                AlignRedLineState zera andando pra frente/tras
    area        numero de pixels vermelhos na mascara (proxy de confianca)
    lost        true quando a reta nao pode ser ajustada, mesmo com fallback

NAO tem logica de FSM aqui -- so' o no de visao (ver ApproachRedLineState/
AlignRedLineState em fase4_itjbx).
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from custom_msgs.msg import LaneDirection
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

# (vx, vy, x0, y0): vetor unitario de direcao + um ponto sobre a reta -- o que
# cv2.fitLine devolve. Representacao independente de orientacao (ao contrario
# de coeficientes de y=f(x), que disparam pra reta vertical).
Line = Tuple[float, float, float, float]


@dataclass
class _FollowerState:
    """Estado do detector que precisa persistir entre frames sucessivos."""

    last_line: Optional[Line] = None
    lost_counter: int = 0
    ema_heading: Optional[float] = None
    ema_offset: Optional[float] = None


class RedLineDetectorNode(Node):
    """Detecta a mangueira vermelha (fase4_itjbx) na camera vertical."""

    def __init__(self):
        super().__init__('red_line_detector')

        self.declare_parameter('image_topic', '/vertical_camera/compressed')
        self.declare_parameter('resize_width', 800)

        # Threshold HSV pra vermelho -- DUAS faixas de matiz (h≈0 e h≈180, ver
        # docstring do modulo). Ajuste pro ambiente/iluminacao real.
        self.declare_parameter('red_h1_min', 0)
        self.declare_parameter('red_h1_max', 10)
        self.declare_parameter('red_h2_min', 170)
        self.declare_parameter('red_h2_max', 180)
        self.declare_parameter('red_s_min', 80)
        self.declare_parameter('red_s_max', 255)
        self.declare_parameter('red_v_min', 40)
        self.declare_parameter('red_v_max', 255)
        # Menor que o do lane_detector (5) de proposito -- ver comentario em
        # _build_red_mask sobre a mangueira ser mais fina que a linha azul.
        self.declare_parameter('morph_kernel_size', 3)

        # Minimo de pixels vermelhos na mascara pra tentar ajustar uma reta --
        # substitui o antigo esquema de regioes/n_regions/min_pixels_per_region.
        self.declare_parameter('min_area_px', 40)

        # Suavizacao temporal + fallback quando a mangueira some
        self.declare_parameter('ema_alpha', 0.35)
        self.declare_parameter('max_lost_frames', 8)

        self.declare_parameter('debug_mask', True)
        self.declare_parameter('debug_image', True)
        self.declare_parameter('debug_mask_overlay', True)

        self.publisher_ = self.create_publisher(LaneDirection, '/red_line_detection', _LANE_QOS)
        self.debug_pub_ = self.create_publisher(CompressedImage, 'red_line_detector/debug/compressed', _DBG_QOS)
        self.mask_pub_ = self.create_publisher(CompressedImage, 'red_line_detector/mask/compressed', _DBG_QOS)
        self.mask_overlay_pub_ = self.create_publisher(
            CompressedImage, 'red_line_detector/mask_overlay/compressed', _DBG_QOS)

        image_topic = self.get_parameter('image_topic').value
        self.subscription = self.create_subscription(
            CompressedImage,
            image_topic,
            self.image_callback,
            _LANE_QOS,
        )

        self.br = CvBridge()
        self._state = _FollowerState()
        self.get_logger().info(f'Red line detector started. Subscribed to {image_topic}')

    def _pub_debug(self, publisher, image, header):
        try:
            msg = self.br.cv2_to_compressed_imgmsg(image)
            msg.header = header
            publisher.publish(msg)
        except Exception as exc:
            self.get_logger().error(f'Failed to publish debug image: {exc}')

    def _build_red_mask(self, frame_bgr):
        h1_min = int(self.get_parameter('red_h1_min').value)
        h1_max = int(self.get_parameter('red_h1_max').value)
        h2_min = int(self.get_parameter('red_h2_min').value)
        h2_max = int(self.get_parameter('red_h2_max').value)
        s_min = int(self.get_parameter('red_s_min').value)
        s_max = int(self.get_parameter('red_s_max').value)
        v_min = int(self.get_parameter('red_v_min').value)
        v_max = int(self.get_parameter('red_v_max').value)
        kernel_size = int(self.get_parameter('morph_kernel_size').value)

        blurred = cv2.GaussianBlur(frame_bgr, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Vermelho fica dividido nas duas pontas do matiz -- uniao das duas faixas.
        lower1 = np.array([h1_min, s_min, v_min], dtype=np.uint8)
        upper1 = np.array([h1_max, s_max, v_max], dtype=np.uint8)
        lower2 = np.array([h2_min, s_min, v_min], dtype=np.uint8)
        upper2 = np.array([h2_max, s_max, v_max], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

        # SEM erode/OPEN de proposito -- a mangueira tem so' 2.5cm de diametro
        # (4x mais fina que a linha azul) e um erode/OPEN podia apagar o traco
        # quase inteiro. CLOSE fecha buracos sem afinar; dilate da' folga extra.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, None, iterations=1)
        return mask

    # -- ajuste de reta, robusto a qualquer orientacao ----------------------

    @staticmethod
    def _fit_line(mask: np.ndarray, min_area: int) -> Optional[Line]:
        """Ajusta uma reta a TODOS os pixels vermelhos (cv2.DIST_HUBER, robusto
        a outlier residual, diferente de minimos quadrados puro).

        Devolve (vx, vy, x0, y0) com vy >= 0 sempre -- normaliza a ambiguidade
        de 180 graus da reta ((vx,vy) e (-vx,-vy) descrevem a mesma reta),
        senao theta oscilaria entre valores opostos sem a mangueira se mexer.
        """
        ys, xs = np.nonzero(mask)
        if len(xs) < min_area:
            return None

        points = np.column_stack((xs, ys)).astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()

        if vy < 0:
            vx, vy = -vx, -vy

        return float(vx), float(vy), float(x0), float(y0)

    @staticmethod
    def _compute_outputs(line: Line, width: int, height: int) -> Tuple[float, float]:
        """Retorna (heading_error, forward_offset).

        heading_error: atan2(vx, vy), nao atan2(vy, vx) -- a ordem dos
        argumentos e' o que troca o eixo de referencia de horizontal pra vertical.

        forward_offset: onde a reta cruza a coluna central, relativo ao centro
        vertical. Perto de uma reta vertical (vx perto de 0) essa
        extrapolacao dispara, entao usa-se y0 direto nesse caso.
        """
        vx, vy, x0, y0 = line
        heading_error = math.atan2(vx, vy)

        if abs(vx) > 0.15:
            t = (width / 2.0 - x0) / vx
            y_center = y0 + t * vy
        else:
            y_center = y0

        forward_offset = y_center - height / 2.0
        return heading_error, forward_offset

    @staticmethod
    def _ema(prev: Optional[float], new: float, alpha: float) -> float:
        if prev is None:
            return new
        return alpha * prev + (1 - alpha) * new

    # -- pipeline completo -------------------------------------------------

    def image_callback(self, msg):
        try:
            # ver o mesmo comentario no lane_detector_node.py sobre 'passthrough' vs 'bgr8'
            frame = self.br.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as exc:
            self.get_logger().error(f'Failed to convert image: {exc}')
            return

        resize_width = int(self.get_parameter('resize_width').value)
        scale = resize_width / float(frame.shape[1])
        frame = cv2.resize(frame, (resize_width, int(frame.shape[0] * scale)))
        height, width = frame.shape[:2]

        mask = self._build_red_mask(frame)
        area = int(np.count_nonzero(mask))

        if bool(self.get_parameter('debug_mask').value):
            mask_dbg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.putText(mask_dbg, f'red px={area}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            self._pub_debug(self.mask_pub_, mask_dbg, msg.header)

        min_area_px = int(self.get_parameter('min_area_px').value)
        ema_alpha = float(self.get_parameter('ema_alpha').value)
        max_lost_frames = int(self.get_parameter('max_lost_frames').value)

        line = self._fit_line(mask, min_area_px)

        if line is None:
            # Throttled pra nao inundar o log a 20Hz -- serve pra diagnosticar se
            # a causa de uma perda de deteccao e' pouco pixel (limiar apertado
            # demais, ou a mangueira genuinamente fora do quadro) ou outra coisa.
            self.get_logger().warn(
                f'Nenhum ajuste possivel: {area}px vermelhos (min_area_px={min_area_px})',
                throttle_duration_sec=1.0)

        state = self._state
        red_msg = LaneDirection()

        used_line: Optional[Line] = None
        status_text = ''

        if line is not None:
            state.last_line = line
            state.lost_counter = 0
            used_line = line

            heading, offset = self._compute_outputs(line, width, height)

            state.ema_heading = self._ema(state.ema_heading, heading, ema_alpha)
            state.ema_offset = self._ema(state.ema_offset, offset, ema_alpha)

            red_msg.lost = False
            red_msg.theta = float(state.ema_heading)
            red_msg.y_centroid = int(round(state.ema_offset))
            status_text = f'heading={state.ema_heading:+.2f} offset={state.ema_offset:+.1f}px'
        else:
            state.lost_counter += 1
            if state.lost_counter <= max_lost_frames and state.last_line is not None:
                # extrapola com o ultimo fit valido por alguns frames (dropout curto)
                used_line = state.last_line
                heading, offset = self._compute_outputs(state.last_line, width, height)

                red_msg.lost = False
                red_msg.theta = float(state.ema_heading if state.ema_heading is not None else heading)
                red_msg.y_centroid = int(round(state.ema_offset if state.ema_offset is not None else offset))
                status_text = f'EXTRAPOLATED (lost {state.lost_counter}/{max_lost_frames})'
            else:
                # perdido de verdade: zera estado
                red_msg.lost = True
                red_msg.theta = 0.0
                red_msg.y_centroid = 0
                state.last_line = None
                status_text = 'LOST'

        if area > 0:
            ys, xs = np.nonzero(mask)
            red_msg.x_centroid = int(round(float(np.mean(xs)) - width / 2.0))
        else:
            red_msg.x_centroid = 0

        red_msg.area = area

        self.publisher_.publish(red_msg)

        if bool(self.get_parameter('debug_image').value):
            output = self._draw_debug(frame, used_line, status_text, red_msg.lost, width, height)
            self._pub_debug(self.debug_pub_, output, msg.header)

        if bool(self.get_parameter('debug_mask_overlay').value):
            mask_overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_overlay = self._draw_debug(mask_overlay, used_line, status_text, red_msg.lost, width, height)
            self._pub_debug(self.mask_overlay_pub_, mask_overlay, msg.header)

    # -- debug view ---------------------------------------------------------

    def _draw_debug(self, frame, line: Optional[Line], status_text, lost, width, height) -> np.ndarray:
        out = frame.copy()

        if line is not None:
            vx, vy, x0, y0 = line
            # Estende a reta por uma diagonal inteira da imagem pra cada lado do
            # ponto (x0,y0) -- comprido o bastante pra cruzar o quadro em
            # qualquer orientacao, inclusive quase vertical.
            length = float(np.hypot(width, height))
            p1 = (int(x0 - vx * length), int(y0 - vy * length))
            p2 = (int(x0 + vx * length), int(y0 + vy * length))
            cv2.line(out, p1, p2, (0, 165, 255), 2)
            cv2.circle(out, (int(x0), int(y0)), 5, (0, 255, 255), -1)

        cv2.line(out, (0, height // 2), (width, height // 2), (255, 255, 255), 1)
        cv2.line(out, (width // 2, 0), (width // 2, height), (255, 255, 255), 1)

        color = (0, 0, 255) if lost else (0, 255, 0)
        cv2.putText(out, status_text or 'LOST', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return out


def main(args=None):
    rclpy.init(args=args)
    node = RedLineDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
