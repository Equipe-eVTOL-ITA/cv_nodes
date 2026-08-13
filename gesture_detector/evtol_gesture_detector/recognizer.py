"""
Wrapper do reconhecedor de gestos do MediaPipe.

Portado de `gesture_classifier/gesture_recognizer.py` da CBR 2025, que
funcionava e nao e o problema. Duas mudancas:

- o contador global de FPS saiu. Ele usava variaveis de modulo (`COUNTER`,
  `FPS`, `START_TIME`) que sao compartilhadas entre instancias e nunca eram
  lidas por ninguem;
- a interpretacao do resultado saiu deste arquivo. Ela vive em `parsing.py`,
  como funcao pura, para poder ser testada sem MediaPipe e sem camera.

O import do mediapipe e adiado para dentro do construtor de proposito: assim
`parsing.py` e os testes podem ser importados numa maquina sem mediapipe
instalado, o que mantem o CI leve.
"""

import cv2


class GestureRecognizer:
    """Reconhece gestos num quadro BGR, em modo LIVE_STREAM."""

    def __init__(self, model: str, num_hands: int,
                 min_hand_detection_confidence: float,
                 min_hand_presence_confidence: float,
                 min_tracking_confidence: float):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self._mp = mp
        self._result_list = []

        base_options = python.BaseOptions(model_asset_path=model)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            result_callback=self._save_result)
        self.recognizer = vision.GestureRecognizer.create_from_options(options)

    def _save_result(self, result, unused_output_image, timestamp_ms: int):
        self._result_list.append(result)

    def recognize(self, frame):
        """
        Submete um quadro e devolve o resultado ANTERIOR, ou None.

        O modo LIVE_STREAM do MediaPipe e assincrono: `recognize_async` entrega
        o resultado por callback. Portanto o que sai daqui esta sempre um quadro
        atrasado em relacao ao que entrou, e devolve None ate o primeiro
        resultado chegar. Quem consome precisa tolerar os dois casos.
        """
        import time

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=rgb_image)
        self.recognizer.recognize_async(mp_image, time.time_ns() // 1_000_000)

        if self._result_list:
            result = self._result_list[0]
            self._result_list.clear()
            return result
        return None

    def close(self):
        self.recognizer.close()
