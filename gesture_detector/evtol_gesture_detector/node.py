"""
No de reconhecimento de gestos, sobre a classe Detector.

Substitui o `gesture_classifier` da CBR 2025. O que a classe base ja resolve, e
que la era feito a mao:

- inscricao em CompressedImage com QoS de sensor;
- CvBridge e conversao para BGR;
- throttle por frequencia (la era `classification_interval = 0.1` fixo);
- publicacao de imagem de debug com limite de taxa, redimensionamento e
  qualidade JPEG configuraveis (la eram DOIS modos, `full_debug_mode` e
  `light_debug_mode`, com contador de quadros proprio).

O que sobra aqui e o que e do dominio: chamar o MediaPipe e publicar as duas
mensagens.
"""

from collections import deque
import os

import cv2
from custom_msgs.msg import Gesture, HandLocation
from detector.detector import Detector
import rclpy
from sensor_msgs.msg import CompressedImage

from .parsing import centroide_da_mao, comando_estavel, gestures_por_mao
from .recognizer import GestureRecognizer


class GestureDetector(Detector):

    def __init__(self):
        super().__init__('gesture_detector')

        self.declare_parameter('num_hands', 2)
        self.declare_parameter('min_hand_detection_confidence', 0.5)
        self.declare_parameter('min_hand_presence_confidence', 0.5)
        self.declare_parameter('min_tracking_confidence', 0.5)

        # Quantas leituras consecutivas iguais um gesto precisa para ser
        # publicado como estavel. Em 2025 este numero estava escrito a mao em
        # tres lugares, enquanto o parametro existia e era ignorado.
        self.declare_parameter('gesture_debounce', 5)

        self.num_hands = int(self.get_parameter('num_hands').value)
        self.debounce = int(self.get_parameter('gesture_debounce').value)

        # O topico segue a regra do ARCHITECTURE.md: /<nome_do_detector>/...
        # Em 2025 eram /gesture/classification e /gesture/hand_location, e o
        # README documentava um terceiro par de nomes que nao existia.
        self.pub_gestures = self.create_publisher(
            Gesture, '/gesture_detector/gestures', 10)
        self.pub_hand = self.create_publisher(
            HandLocation, '/gesture_detector/hand_location', 10)

        self.debug_pub = None
        if self.get_parameter('debug_image').value:
            self.debug_pub = self.create_publisher(
                CompressedImage, '/gesture_detector/debug/compressed', 10)

        # O modelo mora ao lado deste arquivo e e instalado pelo package_data
        # do setup.py. Se ele sumir, falhe AQUI com uma mensagem clara, e nao
        # la dentro do mediapipe.
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'gesture_recognizer.task')
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f'modelo do mediapipe nao encontrado em {model_path}. '
                'Confira package_data em setup.py: sem ele o .task nao e '
                'instalado e o pacote so funciona com --symlink-install.')

        self.recognizer = GestureRecognizer(
            model=model_path,
            num_hands=self.num_hands,
            min_hand_detection_confidence=float(
                self.get_parameter('min_hand_detection_confidence').value),
            min_hand_presence_confidence=float(
                self.get_parameter('min_hand_presence_confidence').value),
            min_tracking_confidence=float(
                self.get_parameter('min_tracking_confidence').value))

        self._historico = deque(maxlen=max(self.debounce, 1))
        self._ultimos_gestos = [''] * self.num_hands
        self._ultimo_centroide = None

        self.get_logger().info(
            f'gesture_detector com {self.num_hands} mao(s), '
            f'debounce de {self.debounce} leituras')
        self.get_logger().info(
            'publicando em /gesture_detector/gestures e '
            '/gesture_detector/hand_location')

    def process_frame(self, frame, header):
        result = self.recognizer.recognize(frame)

        # O modo LIVE_STREAM devolve None ate o primeiro resultado chegar, e a
        # cada quadro em que o callback ainda nao disparou. Isso NAO significa
        # "nenhuma mao": significa "sem resposta nova". Publicar lista vazia
        # aqui faria a missao achar que perdeu a mao a cada outro quadro.
        if result is not None:
            self._ultimos_gestos = gestures_por_mao(result, self.num_hands)
            self._ultimo_centroide = centroide_da_mao(result, 0)

            msg = Gesture()
            msg.gestures = self._ultimos_gestos
            self.pub_gestures.publish(msg)

            # A posicao da mao sai SEMPRE que ha mao, qualquer que seja o gesto.
            # Ver o comentario em parsing.centroide_da_mao.
            if self._ultimo_centroide is not None:
                hand = HandLocation()
                hand.hand_x = float(self._ultimo_centroide[0])
                hand.hand_y = float(self._ultimo_centroide[1])
                self.pub_hand.publish(hand)

            self._historico.append(self._ultimos_gestos[0]
                                   if self._ultimos_gestos else '')

        if self.debug_pub is not None and self._debug_should_publish():
            self._pub_debug(self.debug_pub, self._anotar(frame, result), header)

    def _anotar(self, frame, result):
        """Desenha landmarks e rotulos. So roda quando o debug esta ligado."""
        img = frame.copy()
        landmarks = getattr(result, 'hand_landmarks', None) if result else None
        if not landmarks:
            return img

        altura, largura = img.shape[:2]
        for i, mao in enumerate(landmarks):
            for ponto in mao:
                cv2.circle(img,
                           (int(ponto.x * largura), int(ponto.y * altura)),
                           4, (0, 255, 0), -1)

            if i < len(self._ultimos_gestos) and self._ultimos_gestos[i]:
                x = int(min(p.x for p in mao) * largura)
                y = int(min(p.y for p in mao) * altura)
                cv2.putText(img, f'{i}: {self._ultimos_gestos[i]}',
                            (x, max(y - 10, 20)), cv2.FONT_HERSHEY_DUPLEX,
                            0.7, (255, 255, 255), 2, cv2.LINE_AA)

        estavel = comando_estavel(self._historico, self.debounce)
        if estavel:
            cv2.putText(img, f'estavel: {estavel}', (10, altura - 15),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2,
                        cv2.LINE_AA)
        return img

    def destroy_node(self):
        try:
            self.recognizer.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GestureDetector()
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
