from pathlib import Path
from tarfile import TarFile
from urllib import request

import cv2
import numpy as np


class CeliaLoader:
    '''A caching data loader for the celia segmentation dataset.
    '''

    def __init__(self, cache_dir='./data', url='https://storage.googleapis.com/uga-dsp/project4'):
        '''Initialize a CeliaLoader.

        Args:
            cache_dir: The location of the cache.
            url: The base url of the dataset.
        '''
        cache_dir = Path(cache_dir)
        self.cache_dir = cache_dir
        self.url = url

    def open(self, path, mode='r', force_download=False, **kwargs):
        '''Opens a file in the celia dataset.

        Args:
            path: The path to the file, relative to the base URL.
            mode: The access mode, same as the builtin `open`.
            force_download: Always download; ignore the cache.
            kwargs: Forwarded to `pathlib.Path.open`.
        '''
        path = Path(path)
        local = self.cache_dir / path

        if force_download or not local.exists():
            local.parent.mkdir(parents=True, exist_ok=True)
            url = f'{self.url}/{path}'
            with request.urlopen(url) as response:
                with local.open('wb', **kwargs) as fd:
                    for line in response:
                        fd.write(line)

        return local.open(mode, **kwargs)

    def train(self):
        '''Iterate over training data.

        Yields:
            data: A numpy array for a training video.
            mask: A numpy array for the corresponding mask.
        '''
        with self.open('train.txt') as manifest:
            for name in manifest:
                name = name.strip()  # Strip trailing newline
                datafile = self.open(f'data/{name}.tar', 'rb')
                maskfile = self.open(f'masks/{name}.png', 'rb')

                extraction_dir = self.cache_dir / 'unpacked'
                tar = TarFile(fileobj=datafile)
                tar.extractall(extraction_dir)

                data = sorted(x.name for x in tar)
                data = (f'{extraction_dir}/{name}' for name in data)
                data = (cv2.imread(name, cv2.IMREAD_GRAYSCALE) for name in data)
                data = np.stack(data)

                mask = bytearray(maskfile.read())
                mask = np.asarray(mask, 'uint8')
                mask = cv2.imdecode(mask, cv2.IMREAD_GRAYSCALE)

                yield data, mask

                tar.close()
                datafile.close()
                maskfile.close()

    def test(self):
        '''Iterate over test data.

        Yields:
            data: A numpy array for a test video.
        '''
        with self.open('test.txt') as manifest:
            for name in manifest:
                name = name.strip()  # Strip trailing newline
                datafile = self.open(f'data/{name}.tar', 'rb')

                extraction_dir = self.cache_dir / 'unpacked'
                tar = TarFile(fileobj=datafile)
                tar.extractall(extraction_dir)

                data = sorted(x.name for x in tar)
                data = (f'{extraction_dir}/{name}' for name in data)
                data = (cv2.imread(name, cv2.IMREAD_GRAYSCALE) for name in data)
                data = np.stack(data)

                yield data

                tar.close()
                maskfile.close()
