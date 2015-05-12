# lpBLAS: Limited Precision BLAS Subset

This contains the implementation of 8-bit int and 16-bit int
for a subset of BLAS Level1 procedures. The goal is to make
a single thread eats more than 80% of the total memory bandwidth
(12GB/s) for functions like dot products, random rounding,
and conversions. For now, we only care about functions that
might help the NIPS version of Buckwild!.


### Limitations
  - Currently, only support Haswell.
  - TODO: Working on Sparse domain (May 12 2015)

### Current Performance

    $ ./a.out 
    | Convert (f32->8) = 11216.5 MB/s   t = 0.042512 seconds.
    | Convert (8->f32) = 8589.18 MB/s   t = 0.055516 seconds.
    |    Approximate: -0.20707 -> -0.204724
    |    Approximate: 0.680971 -> 0.677165
    |    Approximate: -0.293328 -> -0.291339
    |    Approximate: -0.106833 -> -0.102362
    |    Approximate: -0.362614 -> -0.362205
    | Dot (8) = 10604.6 MB/s   t = 0.017986 seconds.
    |    Approximate: 4527.75 -> 4631.64
    
    | Convert (f32->16) = 11276.5 MB/s   t = 0.050743 seconds.
    | Convert (16->f32) = 9488.82 MB/s   t = 0.060303 seconds.
    |    Approximate: 0.495487 -> 0.495468
    |    Approximate: 0.528764 -> 0.528764
    |    Approximate: -0.321082 -> -0.321055
    |    Approximate: 0.9764 -> 0.976379
    |    Approximate: -0.934567 -> -0.934538
    | Dot (16) = 12464.7 MB/s   t = 0.030604 seconds.
    |    Approximate: -2828.06 -> -2828.25
    
    | Convert (f32->32) = 13658.6 MB/s   t = 0.055858 seconds.
    | Convert (32->f32) = 11921.5 MB/s   t = 0.063997 seconds.
    |    Approximate: -0.968859 -> -0.968859
    |    Approximate: -0.929235 -> -0.929235
    |    Approximate: 0.164984 -> 0.164984
    |    Approximate: 0.762335 -> 0.762335
    |    Approximate: -0.772878 -> -0.772878
    | Dot (32) = 12588.1 MB/s   t = 0.060608 seconds.
    |    Approximate: 2773.13 -> 2772.8
    
