from setuptools import find_packages, setup

package_name = 'mangueira_detector'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='marconipavan',
    maintainer_email='angelo.marconi.pavan@gmail.com',
    description='Red hose (mangueira) detector for SAE 2026 Mission 2.',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'mangueira_detector_node = mangueira_detector.mangueira_detector_node:main',
        ],
    },
)
