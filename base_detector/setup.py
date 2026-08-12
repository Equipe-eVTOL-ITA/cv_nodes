from setuptools import find_packages, setup

# O nome do PACOTE ROS continua base_detector -- e o que os launches usam.
# O que mudou foi o nome do MODULO Python interno, que era tambem
# `base_detector` e por isso colidia com o modulo de topo de mesmo nome exposto
# pelo itajuba_cv_utils. Ver o comentario no entry_points abaixo.
package_name = 'base_detector'
module_name = 'evtol_base_detector'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Equipe eVTOL ITA',
    maintainer_email='angelo.marconi.pavan@gmail.com',
    description='Detector de bases de pouso por combinacao de azul e amarelo',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            # NAO renomeie `evtol_base_detector` de volta para `base_detector`.
            #
            # O console script mora em install/base_detector/lib/base_detector/
            # e e inequivoco, mas o IMPORT dentro dele resolve pelo PYTHONPATH.
            # Enquanto o modulo se chamava `base_detector`, qualquer outro
            # pacote do workspace que expusesse um modulo de topo com esse nome
            # podia vencer -- e um vencia: o itajuba_cv_utils, cujo diretorio de
            # build entra no PYTHONPATH antes deste pacote.
            #
            # O sintoma nao era erro. Era `ros2 run base_detector base_detector`
            # subindo OUTRO detector, que assina /vertical_camera cru e publica
            # em /vertical_camera/classification. A fase 1, que espera
            # /base_detector/detections, varria a arena inteira sem ver base
            # nenhuma e voltava para casa sem uma mensagem sequer.
            f'{package_name} = {module_name}.node:main'
        ],
    },
)
