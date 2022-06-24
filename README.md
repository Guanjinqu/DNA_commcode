# DNA_commcode
Channel coding for the DNA storage domain (synchronous error channels), including convolutional codes, LDPC, turbo, and other iterative decoding methods.

## Requirements
- tqdm
- matplotlib
- numpy
- pandas
- scikit-commpy
- pyldpc

## Introduction
As this project is still in progress, the code is currently provided for reference only. If you have any questions feel free to contact me :D
- ref2bin.py
> A tool for converting files and DNA sequences to each other.
- test_multiprocessing.py
> Multi-process code 
- test_comm.py
> The main program, after setting the parameters, will automatically generate the required codecs, and the fast_mode function will perform the whole process of "production information - encoding - channel - decoding - error rate statistics".
- test_encode.py
> Encoder
- test_decode.py
> Decoder
- test_channel.py
> Channel
- test_acc.py
> Statistics correct rate
- test_BCJR.py
> Convolutional code decoder based on BCJR algorithm
- test_LDPC.py
> LDPC decoder based on BP algorithm

## License

Clover is licensed under the GNU General Public License, for more information read the LICENSE file or refer to:

http://www.gnu.org/licenses/

## Citation

A related article is being written.
