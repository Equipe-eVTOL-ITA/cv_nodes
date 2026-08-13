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

from custom_msgs.msg import Gesture, HandLocation
import cv2
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
        self._quadros_anotados = 0

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
        """
        Desenha o que o reconhecedor viu. So roda com o debug ligado.

        Esta imagem responde a pergunta 'a visao esta funcionando?', que sem ela
        so da para responder por eliminacao. Ela mostra, nesta ordem de utilidade:

          - se ha QUADRO CHEGANDO (o contador sobe);
          - se ha MAO, e onde;
          - o CENTROIDE, que e o ponto que de fato controla o drone;
          - a MIRA no centro da imagem, que e o alvo dos dois PIDs;
          - o gesto de cada mao e o comando estavel.

        Sem mao, ela diz 'SEM MAO' em vez de simplesmente nao desenhar nada --
        um quadro limpo e ambiguo entre 'nao vejo mao' e 'nao chegou quadro'.
        """
        img = frame.copy()
        altura, largura = img.shape[:2]
        landmarks = getattr(result, 'hand_landmarks', None) if result else None

        # Mira no centro: o setpoint dos PIDs de guinada e de altitude. O drone
        # gira e sobe ate por o centroide da mao em cima dela.
        cx, cy = largura // 2, altura // 2
        cv2.drawMarker(img, (cx, cy), (128, 128, 128), cv2.MARKER_CROSS, 24, 1)

        self._quadros_anotados += 1
        cv2.putText(img, f'quadro {self._quadros_anotados}', (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
                    cv2.LINE_AA)

        if not landmarks:
            cv2.putText(img, 'SEM MAO', (10, altura - 15),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2,
                        cv2.LINE_AA)
            return img

        for i, mao in enumerate(landmarks):
            for ponto in mao:
                cv2.circle(img,
                           (int(ponto.x * largura), int(ponto.y * altura)),
                           4, (0, 255, 0), -1)

            rotulo = (self._ultimos_gestos[i]
                      if i < len(self._ultimos_gestos) and self._ultimos_gestos[i]
                      else '?')
            x = int(min(p.x for p in mao) * largura)
            y = int(min(p.y for p in mao) * altura)
            cv2.putText(img, f'{i}: {rotulo}', (x, max(y - 10, 40)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2,
                        cv2.LINE_AA)

        # O centroide da mao 0 e o unico ponto que sai deste no como posicao, e
        # e o que os PIDs perseguem. Ve-lo na imagem e a forma mais direta de
        # entender por que o drone esta girando para um lado ou para o outro.
        if self._ultimo_centroide is not None:
            px = int(self._ultimo_centroide[0] * largura)
            py = int(self._ultimo_centroide[1] * altura)
            cv2.circle(img, (px, py), 9, (0, 128, 255), 2)
            cv2.line(img, (cx, cy), (px, py), (0, 128, 255), 1)

        estavel = comando_estavel(self._historico, self.debounce)
        cv2.putText(img, f'estavel: {estavel or "-"}', (10, altura - 15),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7,
                    (0, 255, 255) if estavel else (150, 150, 150), 2,
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
