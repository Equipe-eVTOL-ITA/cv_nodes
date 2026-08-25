from glob import glob
import os

from setuptools import find_packages, setup

# O nome do PACOTE ROS e' base_detector_itjbx2026 -- e' o que os launches usam.
# O MODULO Python interno leva o prefixo evtol_ de proposito: o base_detector
# da CBR ja bateu de frente com um modulo de topo homonimo de outro pacote do
# workspace (itajuba_cv_utils), porque o import de dentro do console_script
# resolve pelo PYTHONPATH, nao pelo nome inequivoco do pacote ROS. O sintoma
# nao foi erro -- foi `ros2 run` subindo o detector ERRADO, em silencio. Ver o
# comentario identico em cv_nodes/base_detector/setup.py.
package_name = 'base_detector_itjbx2026'
module_name = 'evtol_base_detector_itjbx2026'

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
    maintainer='Equipe eVTOL ITA',
    maintainer_email='sakonpedro@gmail.com',
    description="Template de detector de bases da missao fase1_itjbx (itajuba_2026)",
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            # NAO renomeie o modulo de volta para base_detector_itjbx2026 --
            # ver o comentario acima.
            f'{package_name} = {module_name}.node:main',
            f'{package_name}_calibrate = {module_name}.calibrate:main',
        ],
    },
)
