from setuptools import find_packages, setup

package_name = 'ransac_window_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ewerton',
    maintainer_email='ewertonaraujofilho@gmail.com',
    description='RANSAC-based window detector simulation node',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'ransac_window_detector = ransac_window_detector.ransac_window_detector:main'
        ],
    },
)