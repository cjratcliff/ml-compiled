from setuptools import setup

setup(
   name='ml-compiled',
   version='0.1',
   description='Quick definitions and intuitive explanations around machine learning',
   author='Chris Ratcliff',
   author_email='c.j.ratcliff@gmail.com',
   packages=['ml-compiled'],  #same as name
    install_requires=[
        'fs (~= 2.0)', 'graphviz (>= 0.8, < 0.9)', 'jinja2 (~= 2.9)', 'mondrian (~= 0.6)', 'packaging (~= 17.1)',
        'psutil (~= 5.4)', 'python-slugify (~= 1.2.0)', 'requests (~= 2.0)', 'stevedore (~= 1.27)', 'whistle (~= 1.0)'
    ],
    extras_require={
        'dev': [
            'cookiecutter (>= 1.5, < 2.2)', 'coverage (>= 4.4, < 5.0)', 'pytest (>= 3.1, < 4.0)',
            'pytest-cov (>= 2.5, < 3.0)', 'pytest-sugar (>= 0.8, < 0.9)', 'pytest-timeout (>= 1, < 2)',
            'sphinx (>= 1.6, < 2.0)', 'sphinx-sitemap (>= 0.2, < 0.3)', 'yapf'
        ],
    },
)
