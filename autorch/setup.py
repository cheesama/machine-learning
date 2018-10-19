from setuptools import setup, find_packages

try:  # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError:  # for pip <= 9.0.3
    from pip.req import parse_requirements

# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = parse_requirements('requirements.txt', session='null')

# reqs is a list of requirement
# e.g. ['django==1.5.1', 'mezzanine==1.4.6']
reqs = [str(ir.req) for ir in install_reqs]

setup(
    name             = 'autorch',
    version          = '0.1',
    description      = 'Machine Learning auto tuning tool for PyTorch',
    author           = 'cheesama',
    author_email     = 'cheehoon12@hanmail.net',
    url              = 'https://github.com/cheesama/autorch',
    install_requires = reqs,
    packages         = find_packages(exclude = ['docs', 'tests*']),
    keywords         = ['pytorch', 'pytorch auto tuning'],
    python_requires  = '>=3',
    zip_safe=False
)
