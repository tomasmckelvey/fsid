# fsid - Frequency Domain Subspace Based Identification
FSID is an open source toolbox, implemented in the Python programming language. 
The toolbox provides scripts which estimates linear multi-input multi-output 
state-space models from sample data using frequency-domain subspace algorithms. 
Algorithms which estimate models based on samples of the transfer function matrix 
as well as frequency domain input and output vectors are provided. The
algorithms can be used for discrete-time models, continuous-time
models as well as for approximation of rational matrices from
samples corresponding to arbitrary points in the complex plane.
To reduce the computational complexity for the estimation
algorithms, an accelerated algorithm is provided which evaluate the
state-space realization of the transfer function matrix at arbitrary
points. The toolbox is compatible with Python 2.7+ as well ass Python 3.0+.
  
The implementation is contained in file python/fsid.py. A the python script python/examples_fsid.py 
illustrates the use of the toolbox. 
