Copyright (c) 2018-2020 R.V. Baluev and D.V. Mikryukov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

    This is the DISTLINK library for a fast, accurate, and reliable computation of the
distance between two confocal elliptic or hyperbolic orbits, or between an ellipse and
a hyperbola.

    The library contains functions intended for calculating the Minimal Orbital
Intersection Distance (MOID) between two confocal bounded Keplerian orbits. A
theoretic method for the computation of the MOID used in the library agorithms
is published in (Kholshevnikov & Vassiliev, 1999; Celest. Mech. Dyn. Astron.,
75, 75-83). The algorithm itself is described in (Baluyev & Mikryukov, 2018;
Astronomy & Computing, 27, 11).

    Additionally to the MOID, this library allows to calculate three different
linking coefficients that describe mutual topology of the orbital configuration.
The linking coefficients can also be used as fast ``surrogates'' of the MOID and
may supply an upper limit on it, see (Kholshevnikov, Vassiliev, 1999; CMDA 75, 67-74).

    The library contains just two source files, the header file 'distlink.h' and
the main file 'distlink.cpp'. They should be compiled together with the user main
program or project.

    An example of compile command for GCC at a high optimization level:

    g++ main.cpp distlink.cpp -O3 -march=native -mfpmath=sse -o main.prg

    Here main.cpp is the file with your main function.
    Please do not use -ffast-math flag, as it causes computing errors.
    The detailed description of the functions and their calling interaface is given
in the comments inside distlink.h. Here we give only a brief general explanaition.

**********************************************************************************

Main data types and functions declared in the file 'distlink.h':


0. The library includes template functions that accept a type parameter 'realfp'.
   It can be 'float', 'double', or 'long double'. Selecting 'float' means to obtain
   the fastest computing, but least accurate. In fact, its practicality is doubtful
   due to increased numeric errors. The recommended type is 'double', which allows
   for quick and accurate computations. The 'long double' type is even more accurate
   but implies significant slowdown. It can be used to reprocess numerically
   difficult almost-degenerate cases.

1. The class 'COrbitData' is designed to store orbital elements. Fields in this
   class contain five orbital elements and auxiliary data, e.g. the basis unit
   vectors P, Q that define orbit orientation.

2. The structure 'SMOIDResult' stores all necessary and auxiliary information
   obtained after computation of the MOID. Its fields contain information
   about the MOID, the corresponding eccentric anomalies, and the reliability
   of the result. The field 'good' is set to false if the algorithm finds that
   its results are not numerically reliable.

3. The structure 'SLCResult' is designed to store two linking coefficients and the
   mutual inclination of the pair of orbits.

4. The function
     detect_suitable_options(...)
   suggests suitable tolerance parameters to be passed to the functions MOID_fast
   and MOID_direct_search. They are selected according to the requested floating
   point arithmetic (realfp type). It sets max_root_error (or delta_max) to square
   root of machine epsilon, and min_root_error (or delta_min) is set to the double
   machine epsilon.

5. The function
     test_peri_apo(O1, O2, limit)
   allow to quickly test, whether two orbits may have MOID below the specified
   limit, or it is definitely above. It is based on a simplistic comparison of
   orbital periceter and apocenter distances.

6. The function
     LC(O1, O2, min_mut_incl)
   computes all the linking coefficients and mutual inclination between the
   specified orbits. It returns a structure of SLCResult type.

7. The function
     MOID_fast(O1, O2, max_root_error, min_root_error, nu)
   is the central function in the library. Its purpose is to calculate the MOID
   between two specified orbits O1 and O2. It returns a structure of SMOIDResult
   type that includes a self-diagnostic boolean value indicating numerical
   reliability of the result. If MOID_fast(O1, O2, ...) fails to yield a reliable
   result then try MOID_fast(O2, O1, ...), and if it fails again then run
   MOID_fast at a higher numeric arithmetic (long double instead of double), and
   if this fails too, run MOID_direct_search(O1, O2, ...).

8. The function
     MOID_direct_search(O1, O2, densities, max_dist_error, max_anom_error)
   should be used when 'MOID_fast' returns unreliable results. It calculates the
   MOID by means of one-dimensional search in one of positional angular variables.
   By default, the eccentric anomaly on the first orbit O1 is scanned; the position
   on the second orbit O2 is calculated analytically. However, the orbits may be
   interchanged internally if the algorithm finds that it allows to reduce the
   angular scan range. The algorithm stops when the scannable eccentric anomaly
   and the minimum distance are both located with the required accuracy.
