import numpy as np


def modulate(data):
    return np.array([1 if bit == 1 else -1 for bit in data])


def demodulate(received_signal):
    return np.array([1 if x > 0 else 0 for x in received_signal])


def add_white_gaussian_noise(signal, snr_db):
    snr_linear = 10 ** (snr_db / 10.0)
    power_signal = np.mean(np.abs(signal) ** 2)
    noise_power = power_signal / snr_linear
    noise = np.sqrt(noise_power / 2) * np.random.randn(len(signal))
    return signal + noise


def transmit(data, snr):
    modulated_data = modulate(data)
    noisy_data = add_white_gaussian_noise(modulated_data, snr)
    demodulated_data = demodulate(noisy_data)

    return demodulated_data, noisy_data
