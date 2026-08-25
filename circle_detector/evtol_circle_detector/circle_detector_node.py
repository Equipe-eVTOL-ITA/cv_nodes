"""
circle_detector_node.py

Acha a base circular azul (marcador de decolagem, mission_line) usando
cv2.HoughCircles e estima por onde a linha sai dela (a base e' redonda mas a
linha so' sai de um lado). Roda separado do lane_detector de proposito: so'
precisa estar ativo no comeco da missao, enquanto a base ainda esta a vista.
Dois jeitos de se encerrar (processo termina de vez, nao so' fica ocioso,
pra nao gastar processamento pelo resto da missao):
  1. Recebe uma mensagem em /circle_detector/shutdown -- o FollowLineState
     publica isso assim que a missao entra em FOLLOW_LINE, ja' que o circulo
     nao serve mais pra nada a partir dai'.
  2. Fallback: 'active_duration_s' segundos desde o start, caso o sinal acima
     nunca chegue (ex: missao nunca sai de SEARCH_LINE e pousa por timeout).
O lane_detector so' escuta a ultima deteccao publicada aqui (ver BaseCircle)
pra recortar o disco da propria mascara; ele nao depende desse no continuar
rodando.

Publica custom_msgs/msg/BaseCircle em /base_circle:
    found       true quando um circulo azul foi encontrado no frame
    x, y        centro do circulo (px, na imagem redimensionada)
    radius      raio do circulo (px)
    exit_valid  true quando foi possivel estimar por onde a linha sai dele
    exit_theta  direcao (rad, atan2(dx, dy) relativo ao centro da imagem --
                mesma convencao do LaneDirection.theta do lane_detector) de
                saida da linha a partir do circulo; so' valido quando
                exit_valid=true
"""

import math
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from custom_msgs.msg import BaseCircle
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Empty


_QOS = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=5,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
)

# reliable + transient_local pra combinar com o publisher do FollowLineState: mesmo se
# o publish la' acontecer antes do DDS terminar de descobrir esse assinante, a mensagem
# fica retida e chega assim que o match acontecer.
_SHUTDOWN_QOS = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)


class CircleDetectorNode(Node):
    """Acha a base circular azul e por onde a linha sai dela, via HoughCircles."""

    def __init__(self):
        super().__init__('circle_detector')

        self.declare_parameter('image_topic', '/vertical_camera/compressed')
        self.declare_parameter('resize_width', 800)

        # Threshold HSV pra azul -- mesma faixa do lane_detector, ajuste junto
        self.declare_parameter('blue_h_min', 100)
        self.declare_parameter('blue_h_max', 130)
        self.declare_parameter('blue_s_min', 80)
        self.declare_parameter('blue_s_max', 255)
        self.declare_parameter('blue_v_min', 40)
        self.declare_parameter('blue_v_max', 255)
        self.declare_parameter('morph_kernel_size', 5)

        # dp=1.0 + blur na mascara antes do Hough (ver image_callback) + param2
        # mais tolerante -- sem isso a deteccao flickava mesmo com o circulo
        # bem visivel, porque o pico do acumulador ficava marginal.
        self.declare_parameter('hough_blur_kernel', 9)
        self.declare_parameter('hough_dp', 1.0)
        self.declare_parameter('hough_min_dist', 200.0)
        self.declare_parameter('hough_param1', 60.0)
        self.declare_parameter('hough_param2', 25.0)
        self.declare_parameter('hough_min_radius', 20)
        self.declare_parameter('hough_max_radius', 0)  # 0 = sem limite superior (convencao do cv2)

        self.declare_parameter('min_exit_pixels', 20)

        # Auto-encerramento: depois desse tempo (s) desde o start, o no' se desliga sozinho.
        self.declare_parameter('active_duration_s', 20.0)

        self.declare_parameter('debug_image', True)

        self.publisher_ = self.create_publisher(BaseCircle, '/base_circle', _QOS)
        self.debug_pub_ = self.create_publisher(CompressedImage, 'circle_detector/debug/compressed', _QOS)

        image_topic = self.get_parameter('image_topic').value
        self.subscription = self.create_subscription(
            CompressedImage, image_topic, self.image_callback, _QOS)

        # Encerra assim que a missao entra em FOLLOW_LINE (ver FollowLineState) -- o
        # circulo nao serve mais pra nada a partir dai'.
        self.shutdown_sub_ = self.create_subscription(
            Empty, '/circle_detector/shutdown', self._on_shutdown_signal, _SHUTDOWN_QOS)

        active_duration_s = float(self.get_parameter('active_duration_s').value)
        self._shutdown_timer = self.create_timer(active_duration_s, self._on_timeout)

        self.br = CvBridge()
        self.get_logger().info(
            f'Circle detector started. Subscribed to {image_topic}, '
            f'auto-shutdown em {active_duration_s:.0f}s (ou antes, se /circle_detector/shutdown chegar)')

    def _on_shutdown_signal(self, _msg: Empty):
        self._shutdown('sinal de /circle_detector/shutdown recebido (entrou em FOLLOW_LINE)')

    def _on_timeout(self):
        self._shutdown('active_duration_s estourou')

    def _shutdown(self, reason: str):
        self.get_logger().info(f'Circle detector: encerrando ({reason}).')
        self._shutdown_timer.cancel()
        rclpy.shutdown()

    def _pub_debug(self, image, header):
        try:
            out_msg = self.br.cv2_to_compressed_imgmsg(image)
            out_msg.header = header
            self.debug_pub_.publish(out_msg)
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

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    @staticmethod
    def _circle_exit_direction(
        mask: np.ndarray, cx: float, cy: float, radius: float, min_exit_pixels: int
    ) -> Tuple[bool, float]:
        """Vetor do CENTRO DO CIRCULO ate' o centroide dos pixels azuis fora do
        raio dele (a linha) -- nao do centro da imagem. Mesma convencao de
        sinal do LaneDirection.theta. Retorna (exit_valid, exit_theta)."""
        ys, xs = np.nonzero(mask)
        dists = np.hypot(xs - cx, ys - cy)
        tail = dists > radius * 1.1

        if not np.any(tail) or int(np.count_nonzero(tail)) < min_exit_pixels:
            return False, 0.0

        target_x = float(np.mean(xs[tail]))
        target_y = float(np.mean(ys[tail]))

        dx = target_x - cx
        dy = cy - target_y
        return True, math.atan2(dx, dy)

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

        mask = self._build_blue_mask(frame)

        dp = float(self.get_parameter('hough_dp').value)
        min_dist = float(self.get_parameter('hough_min_dist').value)
        param1 = float(self.get_parameter('hough_param1').value)
        param2 = float(self.get_parameter('hough_param2').value)
        min_radius = int(self.get_parameter('hough_min_radius').value)
        max_radius = int(self.get_parameter('hough_max_radius').value)
        min_exit_pixels = int(self.get_parameter('min_exit_pixels').value)
        hough_blur_kernel = int(self.get_parameter('hough_blur_kernel').value)

        # Blur leve suaviza a borda dura da mascara binaria numa rampa, dando
        # gradientes mais consistentes ao HoughCircles (que vota em cima do
        # gradiente da imagem).
        hough_input = cv2.GaussianBlur(mask, (hough_blur_kernel, hough_blur_kernel), 0)

        circles = cv2.HoughCircles(
            hough_input, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist,
            param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)

        best: Optional[Tuple[float, float, float]] = None
        if circles is not None:
            # cv2.HoughCircles devolve circles[0] ordenado do voto mais forte no
            # acumulador pro mais fraco -- o primeiro e' a deteccao mais confiavel.
            top = circles[0][0]
            best = (float(top[0]), float(top[1]), float(top[2]))

        out_msg = BaseCircle()
        out_msg.header = msg.header

        if best is not None:
            cx, cy, radius = best
            exit_valid, exit_theta = self._circle_exit_direction(
                mask, cx, cy, radius, min_exit_pixels)
            out_msg.found = True
            out_msg.x = cx
            out_msg.y = cy
            out_msg.radius = radius
            out_msg.exit_valid = exit_valid
            out_msg.exit_theta = float(exit_theta)
        else:
            out_msg.found = False
            out_msg.x = 0.0
            out_msg.y = 0.0
            out_msg.radius = 0.0
            out_msg.exit_valid = False
            out_msg.exit_theta = 0.0

        self.publisher_.publish(out_msg)

        if bool(self.get_parameter('debug_image').value):
            output = frame.copy()
            if best is not None:
                cx, cy, radius = best
                cv2.circle(output, (int(round(cx)), int(round(cy))), int(round(radius)), (0, 255, 0), 2)
                cv2.circle(output, (int(round(cx)), int(round(cy))), 3, (0, 0, 255), -1)
                if out_msg.exit_valid:
                    # seta sai do centro do CIRCULO (nao do centro da imagem) -- e' o
                    # vetor que exit_theta realmente representa agora
                    arrow_len = min(width, height) * 0.4
                    end_x = int(cx + arrow_len * math.sin(out_msg.exit_theta))
                    end_y = int(cy - arrow_len * math.cos(out_msg.exit_theta))
                    cv2.arrowedLine(output, (int(round(cx)), int(round(cy))), (end_x, end_y),
                                     (255, 0, 255), 3, tipLength=0.2)
            status = f'found radius={best[2]:.0f}' if best is not None else 'no circle'
            cv2.putText(output, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            self._pub_debug(output, msg.header)


def main(args=None):
    rclpy.init(args=args)
    node = CircleDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
