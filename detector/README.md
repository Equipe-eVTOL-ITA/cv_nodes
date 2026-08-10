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

## Configuring mission launch files

Every mission that uses a CV node based on `Detector` must follow this pattern in both `simulation.launch.py` and `flight.launch.py`.

### Step 1 — Resolve the detector's YAML path

Each CV package installs its own YAML to its share directory. Load it with `get_package_share_directory` — **never** point to the mission's own YAML for the detector node.

```python
from ament_index_python.packages import get_package_share_directory
import os

# Mission params (FSM node only)
pkg_mission = get_package_share_directory('mission_X')
mission_params = os.path.join(pkg_mission, 'config', 'simulation.yaml')  # or flight.yaml

# Detector params (CV node only) — separate file, separate package
pkg_my_detector = get_package_share_directory('my_detector_package')
detector_params = os.path.join(pkg_my_detector, 'config', 'my_detector.yaml')
```

> **Why separate?** Mission YAMLs are keyed on `mission_X_node:`. A detector node named `my_detector` will not find any matching key there and silently use defaults for every parameter.

### Step 2 — Declare the webcam publisher with an explicit camera_name

The `camera_name` parameter determines the topic name published by `webcam_publisher`:

| `camera_name` | Topic published |
|---|---|
| `'vertical'` | `/vertical_camera/compressed` |
| `'horizontal'` | `/horizontal_camera/compressed` |

Always set it explicitly so the topic is predictable and matches `image_topic` in the detector YAML.

```python
webcam_node = Node(
    package='camera_publisher',
    executable='webcam',
    parameters=[{
        'video_source': '/dev/video2',   # adjust to your device
        'camera_name': 'vertical',       # → /vertical_camera/compressed
    }],
    output='screen',
)
```

For simulation launches that do not start a real camera, omit this node entirely.

### Step 3 — Pass the detector YAML to the CV node

```python
vision_node = Node(
    package='my_detector_package',
    executable='my_detector_node',
    parameters=[detector_params],   # detector YAML, not the mission YAML
    output='screen',
)
```

### Step 4 — Pass only the mission YAML to the FSM node

```python
fsm_node = Node(
    package='mission_X',
    executable='mission_X',
    parameters=[mission_params],    # mission YAML only
    output='screen',
    prefix='nice -n -10',           # higher CPU priority than vision nodes
)
```

### Step 5 — Verify image_topic consistency

Open the detector YAML and confirm that `image_topic` matches the topic the camera publishes:

```yaml
# my_detector_package/config/my_detector.yaml
my_detector:
  ros__parameters:
    image_topic: '/vertical_camera/compressed'   # must match camera_name in webcam_node
```

If they differ the detector will start silently and receive zero frames.

### Complete example

```python
#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    pkg_mission   = get_package_share_directory('mission_X')
    mission_params = os.path.join(pkg_mission, 'config', 'flight.yaml')  # or simulation.yaml

    pkg_detector   = get_package_share_directory('my_detector_package')
    detector_params = os.path.join(pkg_detector, 'config', 'my_detector.yaml')

    webcam_node = Node(
        package='camera_publisher',
        executable='webcam',
        parameters=[{
            'video_source': '/dev/video2',
            'camera_name': 'vertical',
        }],
        output='screen',
    )

    vision_node = Node(
        package='my_detector_package',
        executable='my_detector_node',
        parameters=[detector_params],
        output='screen',
    )

    fsm_node = Node(
        package='mission_X',
        executable=LaunchConfiguration('mission'),
        parameters=[mission_params],
        output='screen',
        prefix='nice -n -10',
    )

    return LaunchDescription([
        DeclareLaunchArgument('mission', default_value='mission_X'),
        webcam_node,
        vision_node,
        TimerAction(period=5.0, actions=[fsm_node]),
    ])
```

### Disabling debug topics at launch time

Debug image topics can be silenced without editing the YAML by overriding parameters inline:

```python
vision_node = Node(
    package='my_detector_package',
    executable='my_detector_node',
    parameters=[
        detector_params,
        {'debug_image': False, 'debug_mask': False},  # override for flight
    ],
    output='screen',
)
```

Or from the command line:
```bash
ros2 launch mission_X flight.launch.py \
  --ros-args -r my_detector_node:debug_image:=false
```

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
