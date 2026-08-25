from glob import glob
import os

from setuptools import find_packages, setup

# O nome do PACOTE ROS e' lane_detector -- e' o que os launches usam.
# O MODULO Python interno leva o prefixo evtol_ de proposito: ver o mesmo
# comentario em cv_nodes/base_detector/setup.py -- ros2 run resolve o import
# pelo PYTHONPATH, nao pelo nome inequivoco do pacote ROS, entao um modulo de
# topo chamado so' 'lane_detector' podia colidir em silencio com outro pacote
# do workspace que reusasse o mesmo nome.
package_name = 'lane_detector'
module_name = 'evtol_lane_detector'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='evtol-ita',
    maintainer_email='evtol.ita@gmail.com',
    description="Segue a linha azul (fase4_itjbx) na câmera vertical via regiões de interesse + regressão polinomial, publica LaneDirection",
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            # NAO renomeie o modulo de volta para lane_detector -- ver comentario acima.
            f'{package_name}_node = {module_name}.lane_detector_node:main',
        ],
    },
)
