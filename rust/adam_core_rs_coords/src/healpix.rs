//! HEALPix `ang2pix` (bead personal-cmy.37.7.1): a faithful port of the
//! `healpix_cxx` `T_Healpix_Base<int64>::ang2pix` / `loc2pix` kernels that
//! back `healpy.ang2pix`, matching the legacy `PointSourceDetections`
//! HEALPix semantics (nested and ring schemes, `lonlat=True` inputs).
//!
//! The floating-point expression order deliberately mirrors the C++ source
//! so pixel assignments agree with healpy, including the near-pole
//! `have_sth` branch selection and the truncating float->int casts.

use std::f64::consts::{FRAC_PI_2, PI};

const TWOTHIRD: f64 = 2.0 / 3.0;
const INV_HALFPI: f64 = 2.0 / PI;

/// healpy `check_nside`: ring accepts any integer in (0, 2^30); nested
/// additionally requires a power of two. The message matches healpy exactly.
pub fn check_nside(nside: i64, nest: bool) -> Result<(), String> {
    let ok = nside > 0 && nside < (1 << 30) && (!nest || (nside & (nside - 1)) == 0);
    if ok {
        Ok(())
    } else {
        Err(format!(
            "{nside} is not a valid nside parameter (must be a power of 2, less than 2**30)"
        ))
    }
}

/// `fmodulo(v1, v2)` from healpix_cxx `math_utils.h`.
fn fmodulo(v1: f64, v2: f64) -> f64 {
    if v1 >= 0.0 {
        if v1 < v2 {
            v1
        } else {
            v1 % v2
        }
    } else {
        let tmp = v1 % v2 + v2;
        if tmp == v2 {
            0.0
        } else {
            tmp
        }
    }
}

/// Spread the low 32 bits of `v` into the even bit positions (Morton).
fn spread_bits(v: i64) -> i64 {
    let mut x = v as u64 & 0xffff_ffff;
    x = (x | (x << 16)) & 0x0000_ffff_0000_ffff;
    x = (x | (x << 8)) & 0x00ff_00ff_00ff_00ff;
    x = (x | (x << 4)) & 0x0f0f_0f0f_0f0f_0f0f;
    x = (x | (x << 2)) & 0x3333_3333_3333_3333;
    x = (x | (x << 1)) & 0x5555_5555_5555_5555;
    x as i64
}

fn xyf2nest(ix: i64, iy: i64, face_num: i64, order: u32) -> i64 {
    (face_num << (2 * order)) + spread_bits(ix) + (spread_bits(iy) << 1)
}

/// `loc2pix` from healpix_cxx `healpix_base.cc`.
fn loc2pix(nside: i64, z: f64, phi: f64, sth: f64, have_sth: bool, nest: bool) -> i64 {
    let nside_f = nside as f64;
    let za = z.abs();
    let tt = fmodulo(phi * INV_HALFPI, 4.0); // in [0,4)

    if !nest {
        // RING scheme.
        let ncap = 2 * nside * (nside - 1);
        let npix = 12 * nside * nside;
        if za <= TWOTHIRD {
            // Equatorial region.
            let nl4 = 4 * nside;
            let temp1 = nside_f * (0.5 + tt);
            let temp2 = nside_f * z * 0.75;
            let jp = (temp1 - temp2) as i64; // ascending edge line
            let jm = (temp1 + temp2) as i64; // descending edge line

            // Ring number counted from z = 2/3, in {1, 2n+1}.
            let ir = nside + 1 + jp - jm;
            let kshift = 1 - (ir & 1); // 1 if ir even, 0 otherwise

            let t1 = jp + jm - nside + kshift + 1 + nl4 + nl4;
            // healpix_cxx uses a mask for power-of-two nside; t1 is always
            // non-negative here so the plain remainder is identical.
            let ip = (t1 >> 1) % nl4; // in {0, 4n-1}
            ncap + (ir - 1) * nl4 + ip
        } else {
            // North and south polar caps.
            let tp = tt - (tt as i64) as f64;
            let tmp = if za < 0.99 || !have_sth {
                nside_f * (3.0 * (1.0 - za)).sqrt()
            } else {
                nside_f * sth / ((1.0 + za) / 3.0).sqrt()
            };

            let jp = (tp * tmp) as i64; // increasing edge line index
            let jm = ((1.0 - tp) * tmp) as i64; // decreasing edge line index

            let ir = jp + jm + 1; // ring number from the closest pole
            let ip = (tt * ir as f64) as i64; // in {0, 4*ir-1}

            if z > 0.0 {
                2 * ir * (ir - 1) + ip
            } else {
                npix - 2 * ir * (ir + 1) + ip
            }
        }
    } else {
        // NEST scheme (nside is a validated power of two).
        let order = nside.trailing_zeros();
        if za <= TWOTHIRD {
            // Equatorial region.
            let temp1 = nside_f * (0.5 + tt);
            let temp2 = nside_f * (z * 0.75);
            let jp = (temp1 - temp2) as i64; // ascending edge line
            let jm = (temp1 + temp2) as i64; // descending edge line

            let ifp = jp >> order; // in {0,4}
            let ifm = jm >> order;
            let face_num = if ifp == ifm {
                ifp | 4
            } else if ifp < ifm {
                ifp
            } else {
                ifm + 8
            };

            let ix = jm & (nside - 1);
            let iy = nside - (jp & (nside - 1)) - 1;
            xyf2nest(ix, iy, face_num, order)
        } else {
            // Polar region, za > 2/3.
            let ntt = std::cmp::min(3, tt as i64);
            let tp = tt - ntt as f64;
            let tmp = if za < 0.99 || !have_sth {
                nside_f * (3.0 * (1.0 - za)).sqrt()
            } else {
                nside_f * sth / ((1.0 + za) / 3.0).sqrt()
            };

            let mut jp = (tp * tmp) as i64; // increasing edge line index
            let mut jm = ((1.0 - tp) * tmp) as i64; // decreasing edge line index
            if jp >= nside {
                jp = nside - 1; // for points too close to the boundary
            }
            if jm >= nside {
                jm = nside - 1;
            }
            if z >= 0.0 {
                xyf2nest(nside - jm - 1, nside - jp - 1, ntt, order)
            } else {
                xyf2nest(jp, jm, ntt + 8, order)
            }
        }
    }
}

/// `ang2pix` from healpix_cxx: colatitude/longitude in radians.
pub fn ang2pix(nside: i64, theta: f64, phi: f64, nest: bool) -> Result<i64, String> {
    if !(0.0..=PI).contains(&theta) {
        return Err("invalid theta value".to_string());
    }
    // Near-pole branch selection matches healpix_cxx `ang2pix` exactly,
    // including its literal `3.14159 - 0.01` threshold (NOT pi - 0.01).
    #[allow(clippy::manual_range_contains, clippy::approx_constant)]
    if theta < 0.01 || theta > 3.14159 - 0.01 {
        Ok(loc2pix(nside, theta.cos(), phi, theta.sin(), true, nest))
    } else {
        Ok(loc2pix(nside, theta.cos(), phi, 0.0, false, nest))
    }
}

/// healpy `ang2pix(..., lonlat=True)`: degrees in, `lonlat2thetaphi`
/// conversion, nside validation, then the C++ kernel.
pub fn ang2pix_lonlat(nside: i64, lon_deg: f64, lat_deg: f64, nest: bool) -> Result<i64, String> {
    check_nside(nside, nest)?;
    let theta = FRAC_PI_2 - lat_deg.to_radians();
    let phi = lon_deg.to_radians();
    ang2pix(nside, theta, phi, nest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nside_validation_matches_healpy() {
        assert!(check_nside(16, true).is_ok());
        assert!(check_nside(3, false).is_ok());
        let err = check_nside(3, true).unwrap_err();
        assert_eq!(
            err,
            "3 is not a valid nside parameter (must be a power of 2, less than 2**30)"
        );
        assert!(check_nside(0, false).is_err());
        assert!(check_nside(1 << 30, false).is_err());
    }

    #[test]
    fn known_pixels_nest_and_ring() {
        // Oracle values from healpy 1.x: healpy.ang2pix(16, lon, lat,
        // nest=..., lonlat=True).
        // Pixel centers of nest pixels 1..5 at nside=16 map back to
        // themselves.
        let centers = [
            (47.8125, 4.780191847199163, 1_i64),
            (42.1875, 4.780191847199163, 2),
            (45.0, 7.180755781458288, 3),
            (50.62499999999999, 7.180755781458288, 4),
            (53.4375, 9.594068226860458, 5),
        ];
        for (lon, lat, pix) in centers {
            assert_eq!(ang2pix_lonlat(16, lon, lat, true).unwrap(), pix);
        }
        // Equator / meridian anchors, both schemes.
        assert_eq!(ang2pix_lonlat(1, 0.0, 0.0, false).unwrap(), 4);
        assert_eq!(ang2pix_lonlat(1, 0.0, 90.0, false).unwrap(), 0);
        assert_eq!(ang2pix_lonlat(1, 0.0, -90.0, false).unwrap(), 8);
    }
}
