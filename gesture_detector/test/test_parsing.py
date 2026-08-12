"""
Testes da interpretacao do resultado do MediaPipe.

Nao importam mediapipe nem abrem camera: os resultados sao dublados. E o que
permite conferir, em milissegundos, a logica que em 2025 so dava para julgar
olhando o drone voar.

Os dois primeiros grupos travam correcoes de defeitos reais de 2025.
"""

from evtol_gesture_detector.parsing import (
    centroide_da_mao,
    comando_estavel,
    gestures_por_mao,
)


# ── Dublês do resultado do MediaPipe ────────────────────────────────────────

class _Categoria:
    def __init__(self, nome, score=0.9):
        self.category_name = nome
        self.score = score


class _Ponto:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class _Resultado:
    def __init__(self, gestures=None, hand_landmarks=None):
        self.gestures = gestures or []
        self.hand_landmarks = hand_landmarks or []


# ── gestures_por_mao ────────────────────────────────────────────────────────

def test_indice_identifica_a_mao_mesmo_com_uma_nao_reconhecida():
    """
    O defeito de 2025: a mao 1 subia para o indice 0.

    A versao antiga filtrava os None, entao um quadro em que so a segunda mao
    e reconhecida produzia uma lista de UM elemento -- e quem lesse o indice 1
    nao via nada, enquanto quem lesse o indice 0 via a mao errada.
    """
    r = _Resultado(gestures=[[], [_Categoria('Pointing_Up')]])
    assert gestures_por_mao(r, 2) == ['', 'Pointing_Up']


def test_lista_tem_sempre_o_tamanho_pedido():
    r = _Resultado(gestures=[[_Categoria('Open_Palm')]])
    assert gestures_por_mao(r, 2) == ['Open_Palm', '']

    vazio = _Resultado()
    assert gestures_por_mao(vazio, 2) == ['', '']
    assert gestures_por_mao(None, 2) == ['', '']


def test_pega_a_categoria_de_maior_score():
    r = _Resultado(gestures=[[_Categoria('Victory', 0.9),
                              _Categoria('ILoveYou', 0.2)]])
    assert gestures_por_mao(r, 1) == ['Victory']


def test_ignora_maos_alem_do_configurado():
    r = _Resultado(gestures=[[_Categoria('A')], [_Categoria('B')],
                             [_Categoria('C')]])
    assert gestures_por_mao(r, 2) == ['A', 'B']


# ── centroide_da_mao ────────────────────────────────────────────────────────

def test_centroide_sai_com_qualquer_gesto():
    """
    O defeito de 2025: a posicao so era publicada com Open_Palm.

    Durante o controle o operador esta apontando ou de punho fechado, entao a
    posicao parava justamente quando o drone precisava segui-la, e os PIDs
    realimentavam o ultimo valor visto.
    """
    r = _Resultado(gestures=[[_Categoria('Closed_Fist')]],
                   hand_landmarks=[[_Ponto(0.4, 0.6), _Ponto(0.6, 0.8)]])
    c = centroide_da_mao(r, 0)
    assert c is not None
    assert abs(c[0] - 0.5) < 1e-9
    assert abs(c[1] - 0.7) < 1e-9


def test_centroide_sem_mao_e_none():
    assert centroide_da_mao(_Resultado(), 0) is None
    assert centroide_da_mao(None, 0) is None
    assert centroide_da_mao(_Resultado(hand_landmarks=[[]]), 0) is None


def test_centroide_de_indice_inexistente_e_none():
    r = _Resultado(hand_landmarks=[[_Ponto(0.5, 0.5)]])
    assert centroide_da_mao(r, 1) is None


# ── comando_estavel (debounce) ──────────────────────────────────────────────

def test_precisa_de_repeticoes_consecutivas():
    assert comando_estavel(['Thumb_Down'] * 5, 5) == 'Thumb_Down'
    assert comando_estavel(['Thumb_Down'] * 4, 5) == ''


def test_ruido_no_meio_derruba_o_comando():
    """Um unico quadro discordante nao pode mandar o drone pousar."""
    historico = ['Thumb_Down', 'Thumb_Down', 'Open_Palm',
                 'Thumb_Down', 'Thumb_Down']
    assert comando_estavel(historico, 5) == ''


def test_olha_so_as_ultimas_leituras():
    """Um comando antigo nao pode ressuscitar por causa do inicio do buffer."""
    historico = ['Open_Palm'] * 5 + ['Victory'] * 3
    assert comando_estavel(historico, 3) == 'Victory'


def test_ausencia_de_gesto_nao_vira_comando():
    assert comando_estavel(['', '', ''], 3) == ''


def test_historico_curto_nao_dispara():
    assert comando_estavel([], 3) == ''
    assert comando_estavel(['Victory'], 3) == ''
