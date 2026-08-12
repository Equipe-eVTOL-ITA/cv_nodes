from setuptools import find_packages, setup

package_name = 'base_detector'

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
            'base_detector = base_detector.base_detector:main'
        ],
    },
)
