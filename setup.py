# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['texture_vae',
 'texture_vae.cli',
 'texture_vae.models',
 'texture_vae.models.tests',
 'texture_vae.utils']

package_data = \
{'': ['*']}

install_requires = \
['click>=8.1.3,<9.0.0',
 'matplotlib>=3.6.2,<4.0.0',
 'pytest>=7.2.0,<8.0.0',
 'setuptools==59.5.0',
 'tensorboard>=2.11.0,<3.0.0',
 'torchsummary>=1.5.1,<2.0.0',
 'tqdm>=4.64.1,<5.0.0']

entry_points = \
{'console_scripts': ['vae-crop-gen = '
                     'texture_vae.cli.crop_gen:create_image_crops']}

setup_kwargs = {
    'name': 'texture-vae',
    'version': '0.1.0',
    'description': '',
    'long_description': '',
    'author': 'Tobias Lang',
    'author_email': 'tobias@motesque.com',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'entry_points': entry_points,
    'python_requires': '>=3.8,<4.0',
}


setup(**setup_kwargs)
