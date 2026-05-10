from setuptools import find_packages, setup

package_name = 'manometro_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    package_data={
        package_name: ['assets/*.png', 'assets/*.svg'],
    },
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='marconipavan',
    maintainer_email='angelo.marconi.pavan@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'manometro_detector = manometro_detector.manometro_detector:main',
            'audio_alert_node   = manometro_detector.audio_alert_node:main',
        ],
    },
)
