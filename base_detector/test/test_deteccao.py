"""Testes da detecção de bases sobre imagens sintéticas.

POR QUE ISTO EXISTE

`colcon build` de um pacote ament_python termina sem NUNCA IMPORTAR o módulo:
ele só instala arquivos. Um erro de sintaxe, um import quebrado ou uma
dependência não declarada passam pelo build inteiro e só aparecem quando alguém
roda o nó — em geral com o Gazebo já de pé e o cronômetro correndo.

O detector antigo deste pacote era um script solto que lia `base.png` do disco e
abria janelas com `cv.imshow`. Não havia como testá-lo sem um humano olhando.
Aqui a base é DESENHADA, então a resposta certa é conhecida por construção.
"""

import cv2
import numpy as np
import pytest
import rclpy
from std_msgs.msg import Header

from base_detector.base_detector import BaseDetector


@pytest.fixture(scope='module')
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def detector(ros):
    node = BaseDetector()
    yield node
    node.destroy_node()


def desenha_base(largura=640, altura=480, centro=(320, 240), lado=200, angulo=0.0):
    """Desenha uma base: quadrado azul com um quadrado amarelo dentro.

    É a geometria real do alvo da CBR — as duas cores adjacentes num quadrado —
    e o que o detector procura.
    """
    img = np.zeros((altura, largura, 3), dtype=np.uint8)
    img[:] = (60, 60, 60)  # fundo cinza, para não casar com nenhuma das cores

    def quadrado(lado_px, cor_bgr):
        rect = ((float(centro[0]), float(centro[1])),
                (float(lado_px), float(lado_px)), angulo)
        cv2.drawContours(img, [cv2.boxPoints(rect).astype(int)], 0, cor_bgr, -1)

    quadrado(lado, (255, 0, 0))              # azul externo
    quadrado(int(lado * 0.5), (0, 255, 255))  # amarelo interno
    return img


def detecta(node, img):
    """Roda o miolo de visão sem passar pelo ROS."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    azul = node._segment(hsv, node.blue_lower, node.blue_upper,
                         node.blue_kernel_size, node.blue_iterations)
    amarelo = node._segment(hsv, node.yellow_lower, node.yellow_upper,
                            node.yellow_kernel_size, node.yellow_iterations)
    dets = node._find_combined_regions(azul, amarelo)
    return node._filter_concentric(dets, img.shape)


def test_encontra_a_base_centrada(detector):
    img = desenha_base()
    dets = detecta(detector, img)

    assert len(dets) == 1, f'esperava uma base, achou {len(dets)}'
    cx, cy = dets[0]['center']
    assert abs(cx - 320) < 15
    assert abs(cy - 240) < 15


def test_nao_inventa_base_em_imagem_vazia(detector):
    img = np.full((480, 640, 3), 60, dtype=np.uint8)
    assert detecta(detector, img) == []


def test_azul_sozinho_nao_e_base(detector):
    """O ponto do detector: uma cor só não basta.

    Um retângulo azul grande — céu, lona, sombra — não pode virar base. É o
    falso positivo que a combinação de duas cores existe para evitar.
    """
    img = np.full((480, 640, 3), 60, dtype=np.uint8)
    cv2.rectangle(img, (200, 150), (440, 330), (255, 0, 0), -1)
    assert detecta(detector, img) == []


def test_amarelo_sozinho_nao_e_base(detector):
    img = np.full((480, 640, 3), 60, dtype=np.uint8)
    cv2.rectangle(img, (200, 150), (440, 330), (0, 255, 255), -1)
    assert detecta(detector, img) == []


def test_duas_bases_separadas_dao_duas_deteccoes(detector):
    img = np.full((480, 640, 3), 60, dtype=np.uint8)
    for cx in (150, 490):
        rect = ((float(cx), 240.0), (140.0, 140.0), 0.0)
        cv2.drawContours(img, [cv2.boxPoints(rect).astype(int)], 0, (255, 0, 0), -1)
        rect = ((float(cx), 240.0), (70.0, 70.0), 0.0)
        cv2.drawContours(img, [cv2.boxPoints(rect).astype(int)], 0, (0, 255, 255), -1)

    dets = detecta(detector, img)
    assert len(dets) == 2, f'esperava duas bases, achou {len(dets)}'


def test_mensagem_sai_normalizada_em_zero_um(detector):
    """O contrato com a vision_geometry: bbox em [0,1], theta em radianos."""
    img = desenha_base(centro=(480, 120), lado=160)
    dets = detecta(detector, img)
    assert len(dets) == 1

    msg = detector._to_msg(dets, img.shape, Header())
    assert len(msg.detections) == 1

    bbox = msg.detections[0].bbox
    assert 0.0 <= bbox.center.position.x <= 1.0
    assert 0.0 <= bbox.center.position.y <= 1.0
    assert 0.0 < bbox.size_x <= 1.0
    assert 0.0 < bbox.size_y <= 1.0

    # Centro em 480/640 = 0.75 e 120/480 = 0.25.
    assert abs(bbox.center.position.x - 0.75) < 0.05
    assert abs(bbox.center.position.y - 0.25) < 0.05

    assert abs(bbox.center.theta) <= np.pi
    assert msg.detections[0].results[0].hypothesis.class_id == 'landing_pad'


def test_normalizacao_e_reversivel(detector):
    """A vision_geometry desfaz a normalização multiplicando de volta.

    Se as duas pontas não concordarem sobre qual dimensão normaliza o quê, o
    erro é de escala e silencioso: a base aparece no lugar certo mas com o
    tamanho errado, e o PnP devolve a altura errada.
    """
    img = desenha_base()
    dets = detecta(detector, img)
    assert len(dets) == 1

    msg = detector._to_msg(dets, img.shape, Header())
    bbox = msg.detections[0].bbox
    altura, largura = img.shape[:2]

    assert abs(bbox.center.position.x * largura - dets[0]['center'][0]) < 1e-6
    assert abs(bbox.center.position.y * altura - dets[0]['center'][1]) < 1e-6
    assert abs(bbox.size_x * largura - dets[0]['size'][0]) < 1e-6
    assert abs(bbox.size_y * altura - dets[0]['size'][1]) < 1e-6
