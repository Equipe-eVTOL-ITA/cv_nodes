from setuptools import find_packages, setup

# O nome do PACOTE ROS e gesture_detector -- e o que os launches usam.
# O MODULO Python interno tem nome proprio (evtol_gesture_detector) porque um
# modulo de topo com nome generico pode ser vencido por outro pacote do
# workspace no PYTHONPATH. Ja aconteceu com o base_detector: o
# itajuba_cv_utils expunha um modulo `base_detector` e o `ros2 run` subia o
# detector errado, sem erro nenhum. Ver cv_nodes v0.2.1.
package_name = 'gesture_detector'
module_name = 'evtol_gesture_detector'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),

    # O MODELO DO MEDIAPIPE PRECISA DISTO. Nao remova.
    #
    # `packages=[...]` instala apenas os .py. O gesture_recognizer.task tem
    # 8 MB e e um arquivo de dados: sem package_data ele NAO e copiado para o
    # install/, e o no procura o modelo ao lado do .py instalado.
    #
    # Medido no pacote gesture_classifier de 2025, que nao tem esta linha:
    #
    #   colcon build                     -> install/.../evtol_gesture_detector/
    #                                       so os .py; modelo ausente
    #   colcon build --symlink-install    -> egg-link aponta para build/, onde o
    #                                       modelo esta; funciona
    #
    # Ou seja: funciona na maquina de desenvolvimento, que usa symlink, e falha
    # no drone, que recebe uma build normal. Exatamente o tipo de diferenca que
    # este workspace existe para eliminar.
    package_data={module_name: ['*.task']},
    include_package_data=True,

    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=False,   # o modelo e lido do disco; nao pode virar zip
    maintainer='Equipe eVTOL ITA',
    maintainer_email='angelo.marconi.pavan@gmail.com',
    description='Reconhecimento de gestos de mao por MediaPipe, sobre a classe Detector',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            f'{package_name} = {module_name}.node:main'
        ],
    },
)
