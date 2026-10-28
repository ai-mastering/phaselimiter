#!/usr/bin/env python3
"""Independent analytic/DFT references, using only Python's 80-digit Decimal."""
import argparse
from decimal import Decimal as D, localcontext
import hashlib
import json
from pathlib import Path
import struct

PI = D('3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679')
ZERO = (D(0), D(0))
DEFAULT_SIZES = (1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 31, 32, 48, 63, 64,
                 127, 255, 256, 257, 1000, 1024, 4095, 4096, 4097, 8192, 16384, 16385, 32768)


def add(a, b):
    return a[0] + b[0], a[1] + b[1]


def mul(a, b):
    return a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]


def cis(angle):
    """Taylor evaluation after argument reduction; independent of libm/FFT."""
    angle %= 2 * PI
    if angle > PI:
        angle -= 2 * PI
    term_c, term_s = D(1), angle
    cosine, sine = term_c, term_s
    for k in range(1, 200):
        term_c *= -angle * angle / ((2 * k - 1) * (2 * k))
        term_s *= -angle * angle / ((2 * k) * (2 * k + 1))
        new_c, new_s = cosine + term_c, sine + term_s
        if new_c == cosine and new_s == sine:
            return new_c, new_s
        cosine, sine = new_c, new_s
    raise ArithmeticError('Decimal trigonometry did not converge')


def powers(n, position, count, inverse=False):
    # Integer reduction avoids introducing a large trigonometric argument.
    root = cis((2 if inverse else -2) * PI * (position % n) / n)
    value = (D(1), D(0))
    for _ in range(count):
        yield value
        value = mul(value, root)


def direct_dft(values):
    n = len(values)
    roots = list(powers(n, 1, n))
    return [tuple(sum((mul(value, roots[(j * k) % n])[part]
                       for j, value in enumerate(values)), D(0)) for part in (0, 1))
            for k in range(n)]


def analytic_signal(n, real, name):
    constant, alternating, impulses = ZERO, ZERO, []
    imag = D(0) if real else D('.0625')
    if name in ('dc', 'mixed'):
        constant = (D('.25'), imag)
    if name in ('nyquist', 'mixed') and n % 2 == 0:
        alternating = (D('.125'), -imag)
    if name.startswith('impulse_'):
        position = {'impulse_start': 0, 'impulse_middle': n // 3, 'impulse_end': n - 1}[name]
        impulses = [(position, (D('.375'), imag))]
    if name == 'mixed':
        impulses = [(n // 3, (D('.375'), imag)), (n - 1, (D('-.0625'), -imag))]
    values = [add(constant, tuple(x * (1 if j % 2 == 0 else -1) for x in alternating))
              for j in range(n)]
    for position, amplitude in impulses:
        values[position] = add(values[position], amplitude)
    count = n // 2 + 1 if real else n
    expected = [ZERO] * count
    expected[0] = tuple(n * x for x in constant)
    if n % 2 == 0:
        expected[n // 2] = add(expected[n // 2], tuple(n * x for x in alternating))
    for position, amplitude in impulses:
        expected = [add(value, mul(amplitude, phase))
                    for value, phase in zip(expected, powers(n, position, count))]
    return values, expected


def independent_inverse(n, real):
    """An independently constructed dyadic spectrum, not a forward result."""
    count = n // 2 + 1 if real else n
    spectrum = [ZERO] * count
    spectrum[0] = (D(n) / 4, D(0) if real else D(n) / 16)
    if n % 2 == 0:
        spectrum[n // 2] = (D(n) / 8, D(0) if real else -D(n) / 32)
    if n >= 3:
        spectrum[1] = (D(n) / 16, D(n) / 32)
    expected = [ZERO] * n
    for k, amplitude in enumerate(spectrum):
        if amplitude == ZERO:
            continue
        for j, phase in enumerate(powers(n, k, n, inverse=True)):
            contribution = mul(amplitude, phase)
            if real and k != 0 and not (n % 2 == 0 and k == n // 2):
                contribution = (2 * contribution[0], D(0))
            expected[j] = add(expected[j], contribution)
    return spectrum, [tuple(x / n for x in value) for value in expected]


def convolution_case(n, real):
    values, _ = analytic_signal(n, real, 'mixed')
    # A sparse kernel keeps the reference O(N), including wraparound at N-1.
    taps = [(0, (D('.5'), D(0))), (1 % n, (D('.25'), D(0))),
            (n // 2, (D('-.125'), D(0) if real else D('.0625'))),
            (n - 1, (D('.0625'), D(0)))]
    kernel = [ZERO] * n
    for position, amplitude in taps:
        kernel[position] = add(kernel[position], amplitude)
    expected = [ZERO] * n
    for i in range(n):
        for position, amplitude in taps:
            expected[i] = add(expected[i], mul(amplitude, values[(i - position) % n]))
    return values + kernel, expected


def sha256(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def generate(destination, sizes=DEFAULT_SIZES):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    with localcontext() as context, (destination / 'inputs.bin').open('wb') as inputs, \
            (destination / 'references.bin').open('wb') as references, \
            (destination / 'cases.tsv').open('w') as manifest:
        context.prec = 80
        manifest.write('id domain n operation input_offset input_count reference_offset output_count reference_kind\n')

        def write_case(name, domain, n, operation, values, expected, kind):
            nonlocal count
            manifest.write(f'{name} {domain} {n} {operation} {inputs.tell() // 16} {len(values)} '
                           f'{references.tell() // 16} {len(expected)} {kind}\n')
            for stream, items in ((inputs, values), (references, expected)):
                for re, im in items:
                    stream.write(struct.pack('<dd', float(re), float(im)))
            count += 1

        for n in sizes:
            if n < 1 or n > 1 << 20:
                raise ValueError('Sizes must be between 1 and 1048576')
            for domain in ('real', 'complex'):
                real = domain == 'real'
                for name in ('zero', 'dc', 'nyquist', 'impulse_start', 'impulse_middle', 'impulse_end', 'mixed'):
                    if name == 'nyquist' and n % 2:
                        continue  # No Nyquist bin exists for odd N; the size is still tested.
                    values, expected = analytic_signal(n, real, name)
                    write_case(name, domain, n, 'forward', values, expected, 'analytic')
                values, _ = analytic_signal(n, real, 'mixed')
                write_case('mixed', domain, n, 'roundtrip', values, values, 'identity')
                spectrum, expected = independent_inverse(n, real)
                write_case('sparse_spectrum', domain, n, 'inverse', spectrum, expected, 'analytic')
                values, expected = convolution_case(n, real)
                write_case('sparse_kernel', domain, n, 'convolution', values, expected, 'direct_convolution')
                if n <= 64:
                    state = 12345
                    values = []
                    for _ in range(n):
                        state = (1664525 * state + 1013904223) & 0xffffffff
                        re = D((state >> 16) % 512 - 256) / 512
                        state = (1664525 * state + 1013904223) & 0xffffffff
                        im = D(0) if real else D((state >> 16) % 512 - 256) / 512
                        values.append((re, im))
                    expected = direct_dft(values)
                    write_case('random', domain, n, 'forward', values,
                               expected[:n // 2 + 1] if real else expected, 'direct_dft')
    metadata = dict(schema=1, decimal_digits=80, reference_storage='little-endian binary64 complex pairs',
                    input_storage='little-endian binary64 complex pairs; dyadic, exact in float32',
                    sizes=list(sizes), cases=count,
                    files={name: sha256(destination / name)
                           for name in ('cases.tsv', 'inputs.bin', 'references.bin')})
    (destination / 'reference.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--sizes', default=','.join(map(str, DEFAULT_SIZES)))
    args = parser.parse_args()
    print(json.dumps(generate(args.destination, tuple(map(int, args.sizes.split(',')))), indent=2))
