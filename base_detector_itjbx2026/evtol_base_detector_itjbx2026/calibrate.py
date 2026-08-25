#!/usr/bin/env python3
"""
Calibracao interativa do base_detector_itjbx2026 -- ajusta os parametros de
deteccao com sliders enquanto olha a mascara ao vivo, em vez de editar YAML e
resubir o no' a cada tentativa.

Uso
---
    ros2 run base_detector_itjbx2026 base_detector_itjbx2026_calibrate

Com a simulacao (ou o drone) publicando `image_topic` (default
/vertical_camera/compressed), abre "camera" (deteccoes desenhadas) e
"mascara" (binario antes do filtro de circularidade). Ajuste ate a mascara
virar um disco branco solido nas bases.

'q' ou Ctrl+C fecha e imprime um bloco pronto para colar em
config/base_detector_itjbx2026.yaml. Usa a MESMA implementacao de deteccao do
no' real (detection.py).
"""

import cv2
from detector.detector import Detector
import numpy as np
import rclpy

from .detection import DetectionParams, detect

_JANELA_CAMERA = 'camera (q para sair e imprimir os parametros)'
_JANELA_MASCARA = 'mascara'


class CalibrationTool(Detector):
    """Reusa Detector so pela assinatura da camera -- nunca publica nada."""

    def __init__(self):
        super().__init__('base_detector_itjbx2026_calibrate')
        self.latest_frame = None

    def process_frame(self, frame: np.ndarray, header) -> None:
        self.latest_frame = frame


def _pct_to_frac(value: int) -> float:
    return value / 100.0


def _noop(_value: int) -> None:
    pass


def _make_trackbars(params: DetectionParams) -> None:
    cv2.namedWindow(_JANELA_MASCARA, cv2.WINDOW_NORMAL)
    w = _JANELA_MASCARA

    # OpenCV trackbars nao tem valor minimo != 0, entao os que precisam
    # comecar em 1 (kernels e iteracoes) sao lidos com +1 na leitura.
    cv2.createTrackbar('white_sat_max', w, params.white_sat_max, 255, _noop)
    cv2.createTrackbar('white_val_min', w, params.white_val_min, 255, _noop)
    cv2.createTrackbar('close_kernel-1', w, params.close_kernel_size - 1, 60, _noop)
    cv2.createTrackbar('close_iters', w, params.close_iterations, 5, _noop)
    cv2.createTrackbar('open_kernel-1', w, params.open_kernel_size - 1, 30, _noop)
    cv2.createTrackbar('open_iters', w, params.open_iterations, 5, _noop)
    cv2.createTrackbar('min_circularity_%', w, int(params.min_circularity * 100), 100, _noop)


def _read_trackbars() -> DetectionParams:
    def g(name: str) -> int:
        return cv2.getTrackbarPos(name, _JANELA_MASCARA)

    return DetectionParams(
        white_sat_max=g('white_sat_max'),
        white_val_min=g('white_val_min'),
        close_kernel_size=g('close_kernel-1') + 1,
        close_iterations=g('close_iters'),
        open_kernel_size=g('open_kernel-1') + 1,
        open_iterations=g('open_iters'),
        min_circularity=_pct_to_frac(g('min_circularity_%')),
        # Area nao entra nos sliders -- ajuste no YAML se precisar.
        min_area_fraction=0.0001,
        max_area_fraction=1.0,
    )


def _print_yaml(params: DetectionParams) -> None:
    print('\n# ---- cole em config/base_detector_itjbx2026.yaml ----')
    print(f'    white_sat_max: {params.white_sat_max}')
    print(f'    white_val_min: {params.white_val_min}')
    print(f'    close_kernel_size: {params.close_kernel_size}')
    print(f'    close_iterations: {params.close_iterations}')
    print(f'    open_kernel_size: {params.open_kernel_size}')
    print(f'    open_iterations: {params.open_iterations}')
    print(f'    min_circularity: {params.min_circularity:.2f}')
    print('# -------------------------------------------------------\n')


def main(args=None):
    rclpy.init(args=args)
    node = CalibrationTool()

    _make_trackbars(DetectionParams())
    last_params = DetectionParams()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)

            if node.latest_frame is None:
                # Sem frame ainda -- provavel que image_topic nao esteja
                # publicando. Ver a checklist no docstring do modulo.
                key = cv2.waitKey(50) & 0xFF
                if key == ord('q'):
                    break
                continue

            last_params = _read_trackbars()
            detections, mask = detect(node.latest_frame, last_params)

            annotated = node.latest_frame.copy()
            for det in detections:
                cx, cy = det['center_px']
                cv2.circle(annotated, (int(cx), int(cy)), int(det['radius']), (0, 255, 0), 2)
                cv2.putText(annotated, f"{det['circularity']:.2f}",
                            (int(cx) - 20, int(cy) - int(det['radius']) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(annotated, f'Bases: {len(detections)}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow(_JANELA_CAMERA, annotated)
            cv2.imshow(_JANELA_MASCARA, mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        _print_yaml(last_params)
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
