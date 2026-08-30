#!/usr/bin/env python3
"""
Detector de bases da missao fase1_itjbx (itajuba_2026).

Encanamento (assinatura de camera, throttle, debug, parametros YAML) herdado
de `detector.detector.Detector` -- ver README de cv_nodes/detector.

Alvo: base quadrada cinza com um circulo BRANCO no meio (a base de decolagem,
redonda e azul sem circulo branco, fica de fora por construcao). A deteccao
procura o circulo branco (mais robusto de segmentar que o cinza sob luz
natural), ignorando a forma preta desenhada dentro dele via CLOSE (e, se
sobrar uma reentrancia mesmo assim, via fallback de hull convexo -- ver
detection.py).

Ja existiu uma fase em que este no' detectava a FORMA preta (triangulo/
hexagono/estrela) dentro do gabarito em vez do circulo -- port de
cv_nodes/RDPformas, testado e funcional, mas a deteccao de circulo se provou
mais robusta na calibracao real e voltou a ser a producao. Aquele codigo
(shape_detection.py, webcam_test_shapes.py) foi tirado do pacote e guardado
em $HOME/shape_detection_itjbx2026 -- nao apagado, caso a equipe queira
retomar a abordagem de forma no futuro.

Este no' publica so' o centro do circulo em pixel NORMALIZADO [0,1] -- a
projecao pra metros (raio do pixel ate' um plano de altura conhecida) fica do
lado C++ (Fase1ItjbxNode::basesCallback + buildCalibration), que ja' tem a
pose do drone sem assinar mais nada.

Contrato de saida (custom_msgs/MultiBaseDetection em <detection_topic>):
publica UMA mensagem por frame, MESMO com `bases` vazia (e' assim que o
assinante sabe que nada esta' visivel NESTE frame -- a acumulacao ao longo da
missao e' responsabilidade do C++). `frame` (JPEG) so' vem preenchido quando
ha' deteccao, pra nao inundar o rosbag. `bases[i].position` e' pixel
normalizado, nao posicao no mundo. `detection_id` e' so' o indice neste
frame, nao um ID persistente.
"""

import cv2
from custom_msgs.msg import BaseDetection, MultiBaseDetection
from detector.detector import Detector
from geometry_msgs.msg import Point
import numpy as np
import rclpy
from sensor_msgs.msg import CompressedImage

from .detection import DetectionParams, detect


class BaseDetectorItjbx2026(Detector):
    """Template de detector de bases -- substitua a deteccao pela sua."""

    def __init__(self):
        # O nome aqui TEM que bater com a primeira chave do YAML
        # (config/flight.yaml ou config/simulation.yaml, conforme o perfil).
        super().__init__('base_detector_itjbx2026')

        # ── Segmentacao do circulo branco ────────────────────────────────
        # Branco = saturacao baixa, valor alto -- sem faixa de matiz (hue
        # nao importa para uma cor sem croma).
        self.declare_parameter('white_sat_max', 60)
        self.declare_parameter('white_val_min', 180)

        # Median blur no HSV antes do threshold -- ver
        # DetectionParams.use_median_blur em detection.py.
        self.declare_parameter('use_median_blur', True)
        self.declare_parameter('median_blur_ksize', 5)

        # Kernel proximo da espessura do traco da forma interna; pequeno
        # demais nao religa reentrancias na borda, grande demais funde bases
        # proximas. Ajuste olhando debug_mask.
        self.declare_parameter('close_kernel_size', 15)
        self.declare_parameter('close_iterations', 2)
        # OPEN pequeno depois, so para tirar ruido residual (nao para lidar
        # com a forma interna -- essa e' tarefa do CLOSE acima).
        self.declare_parameter('open_kernel_size', 3)
        self.declare_parameter('open_iterations', 1)

        # ── Filtro de forma ───────────────────────────────────────────────
        # circularity = 4*pi*area / perimetro^2; 1.0 = circulo perfeito.
        # Comece tolerante (a forma interna ainda deixa arestas residuais
        # mesmo depois do CLOSE) e aperte so se estiver pegando lixo.
        self.declare_parameter('min_circularity', 0.75)
        self.declare_parameter('min_area_fraction', 0.001)  # fracao da imagem
        self.declare_parameter('max_area_fraction', 0.25)

        # Menor angulo interno (graus) tolerado no poligono aproximado do
        # contorno -- complementa a circularidade contra formas com uma
        # ponta saindo (ver DetectionParams em detection.py).
        self.declare_parameter('min_vertex_angle_deg', 30.0)

        # Um quadrado tem circularidade ~0.785 e cantos de 90 -- passa fácil
        # nos dois filtros acima. Rejeita explicitamente quem aproxima para
        # exatamente 4 vertices (contorno cru ou hull).
        self.declare_parameter('reject_quadrilaterals', True)

        # Hull convexo salva contorno com uma reentrancia (mascara que nao
        # fechou 100% num ponto) que falhou em circularidade/angulo -- ver
        # DetectionParams.use_hull_fallback.
        self.declare_parameter('use_hull_fallback', True)

        # ── Refino por Hough (cv2.HoughCircles) ──────────────────────────
        # So' roda no recorte de quem ja passou nos filtros acima -- ver
        # DetectionParams em detection.py.
        self.declare_parameter('use_hough_refine', True)
        self.declare_parameter('hough_dp', 1.0)
        self.declare_parameter('hough_param1', 80.0)
        self.declare_parameter('hough_param2', 20.0)
        self.declare_parameter('hough_radius_margin', 0.3)

        self.declare_parameter('detection_topic', '/base_detector_itjbx2026/detections')

        # Qualidade JPEG do frame embutido em MultiBaseDetection.frame --
        # separada de debug_jpeg_quality de proposito. O debug e' descartavel
        # (throttled, so para acompanhar ao vivo); este frame vira a FOTO
        # salva em disco de cada base, entao vale mais qualidade.
        self.declare_parameter('frame_jpeg_quality', 90)

        detection_topic = self.get_parameter('detection_topic').value
        self.detection_pub = self.create_publisher(MultiBaseDetection, detection_topic, 10)
        self.debug_pub = self.create_publisher(
            CompressedImage, '/base_detector_itjbx2026/debug/bbox/compressed', 1)
        self.mask_debug_pub = self.create_publisher(
            CompressedImage, '/base_detector_itjbx2026/debug/mask/compressed', 1)

        self.get_logger().info(f'base_detector_itjbx2026 publicando em {detection_topic}')

    # ── Interface do Detector ─────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray, header) -> None:
        detections, mask = self._detect(frame)

        self.detection_pub.publish(self._to_msg(detections, frame, header))

        if not self._debug_should_publish():
            return

        if bool(self.get_parameter('debug_image').value):
            annotated = frame.copy()
            for det in detections:
                cx, cy = det['center_px']
                cv2.circle(annotated, (int(cx), int(cy)), int(det['radius']), (0, 255, 0), 2)
                cv2.putText(annotated, f"{det['circularity']:.2f}",
                            (int(cx) - 20, int(cy) - int(det['radius']) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(annotated, f'Bases: {len(detections)}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            self._pub_debug(self.debug_pub, annotated, header)

        if bool(self.get_parameter('debug_mask').value):
            self._pub_debug(self.mask_debug_pub, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), header)

    # ── Deteccao ────────────────────────────────────────────────────────

    def _params(self) -> DetectionParams:
        """Le os parametros ROS atuais -- assim um `ros2 param set` em tempo
        real (por exemplo, durante uma calibracao) tem efeito no proximo
        frame, sem reiniciar o no."""
        return DetectionParams(
            white_sat_max=int(self.get_parameter('white_sat_max').value),
            white_val_min=int(self.get_parameter('white_val_min').value),
            use_median_blur=bool(self.get_parameter('use_median_blur').value),
            median_blur_ksize=int(self.get_parameter('median_blur_ksize').value),
            close_kernel_size=int(self.get_parameter('close_kernel_size').value),
            close_iterations=int(self.get_parameter('close_iterations').value),
            open_kernel_size=int(self.get_parameter('open_kernel_size').value),
            open_iterations=int(self.get_parameter('open_iterations').value),
            min_circularity=float(self.get_parameter('min_circularity').value),
            min_area_fraction=float(self.get_parameter('min_area_fraction').value),
            max_area_fraction=float(self.get_parameter('max_area_fraction').value),
            min_vertex_angle_deg=float(self.get_parameter('min_vertex_angle_deg').value),
            reject_quadrilaterals=bool(self.get_parameter('reject_quadrilaterals').value),
            use_hull_fallback=bool(self.get_parameter('use_hull_fallback').value),
            use_hough_refine=bool(self.get_parameter('use_hough_refine').value),
            hough_dp=float(self.get_parameter('hough_dp').value),
            hough_param1=float(self.get_parameter('hough_param1').value),
            hough_param2=float(self.get_parameter('hough_param2').value),
            hough_radius_margin=float(self.get_parameter('hough_radius_margin').value),
        )

    def _detect(self, frame: np.ndarray):
        """
        Acha o circulo branco de cada base (ver o docstring do modulo e
        detection.py, que tem a implementacao de verdade -- compartilhada
        com webcam_test.py).

        Devolve (detections, mask) -- a mascara e devolvida so para o debug
        publicar (`debug_mask: true` no YAML), tunar o CLOSE olhando ela e
        muito mais direto do que olhando o resultado final.
        """
        return detect(frame, self._params())

    # ── Publicacao ────────────────────────────────────────────────────────

    def _to_msg(self, detections: list, frame: np.ndarray, header) -> MultiBaseDetection:
        msg = MultiBaseDetection()

        if detections:
            quality = int(self.get_parameter('frame_jpeg_quality').value)
            ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if ok:
                msg.frame.header = header
                msg.frame.format = 'jpeg'
                msg.frame.data = buf.tobytes()
            else:
                self.get_logger().warning('Falha ao codificar o frame para MultiBaseDetection')

        for i, det in enumerate(detections):
            b = BaseDetection()
            b.header = header

            # Pixel normalizado [0,1], NAO metros -- a projecao para o mundo
            # e feita do lado C++ (ver docstring do modulo).
            cx_norm, cy_norm = det['center_norm']
            b.position = Point(x=cx_norm, y=cy_norm, z=0.0)

            b.base_type = 'estimate'
            b.confidence = min(1.0, det['circularity'])
            b.detection_id = i

            msg.bases.append(b)

        return msg


def main(args=None):
    rclpy.init(args=args)
    node = BaseDetectorItjbx2026()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
