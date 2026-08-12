"""
Interpretacao do resultado do MediaPipe: funcoes puras, testaveis sem camera.

Esta separacao existe pelo mesmo motivo que a `vision_geometry`: a logica que
decide QUAL gesto foi visto e ONDE a mao esta e a que causa comportamento
estranho em voo, e ela nao pode depender de subir camera e mediapipe para ser
conferida.

Duas correcoes sobre a versao de 2025 estao aqui, e ambas mudam comportamento.
Leia antes de mexer.
"""

from typing import List, Optional, Sequence, Tuple


def gestures_por_mao(result, num_hands: int) -> List[str]:
    """
    Nome do gesto de cada mao, com posicao FIXA na lista.

    CORRECAO DE 2025, e e a mais importante deste arquivo.

    A versao antiga montava a lista assim:

        gestures = [g.category_name for g in last_gestures if g is not None]

    O filtro de `None` faz o INDICE DEIXAR DE IDENTIFICAR A MAO. Se a mao 0 nao
    for reconhecida num quadro e a mao 1 sim, a mao 1 aparece no indice 0. No
    quadro seguinte, com as duas reconhecidas, ela volta para o indice 1.

    Isso explica a incoerencia entre os estados da fase 3 de 2025: o
    `SearchState` lia `gestures[0]` e o `GestureControlState` lia `gestures[1]`,
    e nenhum dos dois podia estar certo o tempo todo. Com uma mao so no quadro,
    o controle direcional simplesmente nao respondia.

    Aqui a lista tem SEMPRE `num_hands` posicoes, e mao sem gesto reconhecido
    vira string vazia. O indice passa a significar a mao, sempre.
    """
    saida = [""] * num_hands
    if result is None or not getattr(result, "gestures", None):
        return saida

    for i, categorias in enumerate(result.gestures[:num_hands]):
        if categorias:
            saida[i] = categorias[0].category_name
    return saida


def centroide_da_mao(result, indice: int = 0) -> Optional[Tuple[float, float]]:
    """
    Centroide normalizado dos landmarks de uma mao, ou None se ela nao existe.

    CORRECAO DE 2025. A versao antiga so publicava a posicao da mao quando o
    gesto era `Open_Palm`:

        if "Open_Palm" in gestures_msg.gestures and len(landmarks) > 0:

    Durante o controle por gestos o operador nao esta com a palma aberta: esta
    apontando, fechando o punho, fazendo V. Ou seja, exatamente enquanto o drone
    precisa seguir a mao, a posicao parava de ser publicada, e os PIDs de guinada
    e altitude ficavam realimentando o ULTIMO valor recebido. O drone continuava
    corrigindo em direcao a onde a mao estava da ultima vez que a palma foi
    aberta.

    Aqui a posicao sai sempre que ha landmarks, qualquer que seja o gesto.
    """
    landmarks = getattr(result, "hand_landmarks", None) if result else None
    if not landmarks or indice >= len(landmarks):
        return None

    mao = landmarks[indice]
    if not mao:
        return None

    return (
        sum(p.x for p in mao) / len(mao),
        sum(p.y for p in mao) / len(mao),
    )


def comando_estavel(historico: Sequence[str], minimo: int) -> str:
    """
    Gesto repetido nas ultimas `minimo` leituras, ou vazio se nao houver.

    Debounce: a classificacao oscila entre quadros, e um unico quadro com
    `Thumb_Down` nao pode mandar o drone pousar.

    Em 2025 isto era `allElementsEqual`, que exigia o buffer INTEIRO igual e
    tinha o tamanho 10 escrito a mao em tres lugares, enquanto o parametro
    `gesture_buffer_size` era lido da blackboard e nunca usado.
    """
    if minimo <= 0 or len(historico) < minimo:
        return ""

    ultimos = list(historico)[-minimo:]
    primeiro = ultimos[0]
    if not primeiro:
        return ""
    return primeiro if all(g == primeiro for g in ultimos) else ""
