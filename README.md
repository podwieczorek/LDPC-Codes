# LDPC-Codes

## Table of contents
* [Overview](#Overview)
* [Technologies](#Technologies)
* [Usage](#Usage)
* [Bibliography](#Bibliography)

## Overview
Implementation of LDPC encoders and decoders with benchmark and helper scripts.
Encoders:
* Multiplying message by generator matrix ([encoder.py](encoder.py))
* Back substitution encoder ([bs_encoder.py](bs_encoder.py))

Decoders:
* Bit flipping decoder ([bf_decoder.py](bf_decoder.py))
* Weighted bit flipping decoder ([wbf_decoder.py](wbf_decoder.py))
* Min-sum decoder ([ms_decoder.py](ms_decoder.py))

Program that generates h matrices, implementation of method proposed by Robert Gallager in [1]
– [gallager.py](code_generation/gallager.py).

Helper scripts that convert h matirces from one format to another can be found in [utils](utils)

## Technologies
* matplotlib 3.10.0
* networkx 3.4.2
* numpy 1.23.1

## Usage
Firstly, install requirements:
```
pip install -r requirements.txt
```
Then, simply run main function. Program will create random massages, encode them, 
send them through AWGN BPSK channel and finally decode. Graph of 
BER and FER vs E<sub>b</sub>/ N<sub>0</sub> will be shown.

## Bibliography
[1] R. Gallager, "Low-density parity-check codes," in IRE Transactions on Information Theory, 
vol. 8, no. 1, pp. 21-28, January 1962.

[2] T. J. Richardson and R. L. Urbanke, "Efficient encoding of low-density parity-check codes," 
in IEEE Transactions on Information Theory, vol. 47, no. 2, pp. 638-656, Feb 2001
