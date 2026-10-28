"""Independent 80-digit references for the original true-peak impulse case."""
from decimal import Decimal as D, localcontext
import json
from pathlib import Path
import struct

from fixtures import ZERO, PI, add, mul, powers, cis, sha256

FFT_SIZE = 16384
FIR_SIZE = 4097


def write_values(path, values):
    with Path(path).open('wb') as stream:
        for re, im in values:
            stream.write(struct.pack('<dd', float(re), float(im)))


def read_values(path):
    return [(D.from_float(re), D.from_float(im))
            for re, im in struct.iter_unpack('<dd', Path(path).read_bytes())]


def bessel_i0(x):
    term = result = D(1)
    for k in range(1, 300):
        term *= (x / 2) ** 2 / (k * k)
        updated = result + term
        if updated == result:
            return result
        result = updated
    raise ArithmeticError('I0 did not converge')


def ideal_fir():
    # sinc low-pass with cutoff 1/8 cycles/sample, Kaiser beta = pi * 7.
    center = (FIR_SIZE - 1) // 2
    denominator = bessel_i0(PI * 7)
    coefficients = [ZERO] * FIR_SIZE
    coefficients[center] = (D(1) / 4, D(0))
    for k in range(1, center + 1):
        if k % 4 == 0:
            value = D(0)
        else:
            sine = cis(PI * k / 4)[1]
            window = bessel_i0(PI * 7 * (1 - (D(k) / center) ** 2).sqrt()) / denominator
            value = sine / (PI * k) * window
        coefficients[center - k] = coefficients[center + k] = (value, D(0))
    return coefficients


def decimal_fft(values):
    """Independent radix-2 reference, checked against direct DFT below."""
    n = len(values)
    if n < 1 or n & (n - 1):
        raise ValueError('Reference FFT requires a power-of-two length')
    result = list(values)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            result[i], result[j] = result[j], result[i]
    length = 2
    while length <= n:
        twiddles = list(powers(length, 1, length // 2))
        for start in range(0, n, length):
            for k, twiddle in enumerate(twiddles):
                a = result[start + k]
                b = mul(result[start + k + length // 2], twiddle)
                result[start + k] = add(a, b)
                result[start + k + length // 2] = (a[0] - b[0], a[1] - b[1])
        length *= 2
    return result


def generate_common(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    stamp = destination / 'reference.json'
    source_hash = sha256(__file__)
    if stamp.exists() and json.loads(stamp.read_text()).get('source_sha256') == source_hash:
        return
    with localcontext() as context:
        context.prec = 80
        write_values(destination / 'fir_ideal.bin', ideal_fir())
        roots = list(powers(FFT_SIZE, 1, FFT_SIZE, inverse=True))
        for k, value in ((0, (1, 0)), (FFT_SIZE // 4, (0, 1)),
                         (FFT_SIZE // 2, (-1, 0)), (3 * FFT_SIZE // 4, (0, -1))):
            roots[k] = value  # Exactly known axes; no recurrence residue at endpoints.
        write_values(destination / 'roots.bin', roots)
    stamp.write_text(json.dumps(dict(decimal_digits=80, fft_size=FFT_SIZE, fir_size=FIR_SIZE,
                                    source_sha256=source_hash), indent=2) + '\n')


def refine_kernel(directory, filter_name):
    directory = Path(directory)
    source = directory / f'fir_{filter_name}.bin'
    destination = directory / f'reference_{filter_name}.bin'
    with localcontext() as context:
        context.prec = 80
        coefficients = read_values(source)
        spectrum = decimal_fft(coefficients + [ZERO] * (FFT_SIZE - len(coefficients)))
        spectrum[0] = (spectrum[0][0], D(0))
        spectrum[FFT_SIZE // 2] = (spectrum[FFT_SIZE // 2][0], D(0))
        largest = D(0)
        # Direct DFT checks do not reuse the reference FFT's butterfly layout.
        for k in (0, 1, 2, 4095, 4096, 8191, 8192):
            direct = ZERO
            for value, phase in zip(coefficients, powers(FFT_SIZE, k, len(coefficients))):
                direct = add(direct, mul(value, phase))
            largest = max(largest, abs(direct[0] - spectrum[k][0]), abs(direct[1] - spectrum[k][1]))
        if largest > D('1e-65'):
            raise ArithmeticError(f'Reference FFT/direct DFT disagree: {largest}')
        write_values(destination, [(re / FFT_SIZE, im / FFT_SIZE)
                                   for re, im in spectrum[:FFT_SIZE // 2 + 1]])
    return dict(filter=filter_name, coefficient_sha256=sha256(source), decimal_digits=80,
                direct_dft_spot_check_max=float(largest), reference_sha256=sha256(destination))
