# detector

Base class for ROS2 computer-vision detector nodes.

---

## How to use

### 1. Add `detector` as a dependency

In your package's `package.xml`:
```xml
<exec_depend>detector</exec_depend>
```

In `setup.py` (ament_python packages):
```python
install_requires=['setuptools'],
```
No extra entry needed — `detector` is a plain Python package imported at runtime.

### 2. Subclass `Detector`

```python
from detector.detector import Detector
import numpy as np

class MyDetector(Detector):
    def __init__(self):
        super().__init__('my_detector')   # node name — must match YAML key
        # declare your own parameters here
        self.declare_parameter('my_threshold', 0.5)
        # create your publishers here
        self.result_pub = self.create_publisher(...)

    def process_frame(self, frame: np.ndarray, header) -> None:
        # your detection logic here
        ...
        # publish debug images (throttled, respects debug_image param)
        if bool(self.get_parameter('debug_image').value) and self._debug_should_publish():
            self._pub_debug(self.debug_pub_, annotated_frame, header)
```

### 3. Copy and fill the YAML template

```bash
cp src/cv_nodes/detector/config/detector_template.yaml \
   src/my_package/config/my_detector.yaml
```

Replace `nome_do_no` with your node name and tune the values:

```yaml
my_detector:
  ros__parameters:
    image_topic: '/vertical_camera/compressed'
    processing_frequency: 10.0
    debug_image: true
    debug_mask: false
    debug_publish_rate: 5.0
    debug_max_width: 320
    debug_jpeg_quality: 60
    # your own parameters below
    my_threshold: 0.5
```

### 4. Load the YAML in your launch file

```python
import os
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

pkg = get_package_share_directory('my_package')

Node(
    package='my_package',
    executable='my_detector_node',
    parameters=[os.path.join(pkg, 'config', 'my_detector.yaml')],
    output='screen',
)
```

### 5. Switching cameras

Set `image_topic` in the YAML (or override at runtime):

```yaml
# downward camera (default)
image_topic: '/vertical_camera/compressed'

# forward camera
image_topic: '/horizontal_camera/compressed'
```

Or override without changing the YAML:
```bash
ros2 run my_package my_detector_node \
  --ros-args -p image_topic:=/horizontal_camera/compressed
```

---

## How the package works

### Processing-rate throttle

The base class subscribes to the camera topic with `BEST_EFFORT` / `VOLATILE` QoS (compatible with `webcam_publisher`). Every incoming frame passes through `_image_callback_base`, which compares the elapsed time since the last processed frame against `1 / processing_frequency`. Frames that arrive too soon are dropped before any decoding happens, keeping CPU usage bounded regardless of the camera's publish rate.

```
camera topic → _image_callback_base → [rate gate] → process_frame()
                                           ↑
                                   drops frame if too early
```

### Debug-image pipeline

Two helpers are provided so subclasses don't need to re-implement throttling and compression:

| Helper | What it does |
|---|---|
| `_debug_should_publish()` | Returns `True` at most `debug_publish_rate` times per second. Call once per frame and cache the result. |
| `_pub_debug(publisher, image, header)` | Resizes image to `debug_max_width` (if set), JPEG-encodes at `debug_jpeg_quality`, and publishes a `CompressedImage`. |

The `debug_image` and `debug_mask` booleans are exposed as parameters so operators can disable debug topics in flight from a YAML or launch argument without recompiling.

### QoS

| Parameter | Value |
|---|---|
| Reliability | `BEST_EFFORT` |
| Durability | `VOLATILE` |
| History | `KEEP_LAST` (depth 5) |

This matches the `webcam_publisher` QoS profile. A `RELIABLE` subscriber will not receive any frames from a `BEST_EFFORT` publisher — always use `BEST_EFFORT` on both sides.

---

## Example: orange circle detector

```python
import cv2
import numpy as np
import rclpy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import CompressedImage
from detector.detector import Detector


_DBG_QOS = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST, depth=5,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
)


class OrangeCircleDetector(Detector):
    def __init__(self):
        super().__init__('orange_circle_detector')

        self.declare_parameter('h_min', 5)
        self.declare_parameter('h_max', 30)
        self.declare_parameter('s_min', 80)
        self.declare_parameter('v_min', 50)

        self.debug_pub = self.create_publisher(
            CompressedImage, 'orange_circle/debug/compressed', _DBG_QOS)

    def process_frame(self, frame: np.ndarray, header) -> None:
        h_min = int(self.get_parameter('h_min').value)
        h_max = int(self.get_parameter('h_max').value)
        s_min = int(self.get_parameter('s_min').value)
        v_min = int(self.get_parameter('v_min').value)

        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,
                           np.array([h_min, s_min, v_min]),
                           np.array([h_max, 255,   255  ]))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        annotated = frame.copy()
        for cnt in contours:
            if cv2.contourArea(cnt) < 200:
                continue
            (x, y), r = cv2.minEnclosingCircle(cnt)
            cv2.circle(annotated, (int(x), int(y)), int(r), (0, 255, 0), 2)

        if bool(self.get_parameter('debug_image').value) and self._debug_should_publish():
            self._pub_debug(self.debug_pub, annotated, header)


def main(args=None):
    rclpy.init(args=args)
    node = OrangeCircleDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
```

Corresponding YAML (`config/orange_circle_detector.yaml`):

```yaml
orange_circle_detector:
  ros__parameters:
    image_topic: '/horizontal_camera/compressed'
    processing_frequency: 15.0
    debug_image: true
    debug_mask: false
    debug_publish_rate: 5.0
    debug_max_width: 320
    debug_jpeg_quality: 60
    h_min: 5
    h_max: 30
    s_min: 80
    v_min: 50
```
