"""Emit exact upstream versions for pip-audit without installing CUDA wheels.

PyTorch's +cuXXX labels select a binary build. Vulnerability databases track
its upstream release version, so normalize those labels for auditing only.
The installable lock file is left unchanged, and no package is omitted.
"""
import argparse
from pathlib import Path
import re

PIN = re.compile(r'([A-Za-z0-9][A-Za-z0-9_.-]*)==([0-9][A-Za-z0-9.!+_-]*)')


def normalize_requirements(text):
    result = []
    for number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        match = PIN.fullmatch(line)
        if match is None:
            raise ValueError(f'Line {number} must pin one package with ==: {line}')
        name, version = match.groups()
        if name.lower() in {'torch', 'torchvision'}:
            version = version.split('+', 1)[0]
        result.append(f'{name}=={version}')
    if not result:
        raise ValueError('The lock file contains no pinned dependencies')
    return '\n'.join(result) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('lockfile', nargs='?', type=Path, default=Path('requirements-lock.txt'))
    args = parser.parse_args()
    try:
        print(normalize_requirements(args.lockfile.read_text()), end='')
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
