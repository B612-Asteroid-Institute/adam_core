pub mod gauss_iod;
pub use gauss_iod::gauss_iod_fused;

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(v: [f64; 3], s: f64) -> [f64; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

pub fn calc_gibbs_row(r1: [f64; 3], r2: [f64; 3], r3: [f64; 3], mu: f64) -> [f64; 3] {
    let r1_mag = norm(r1);
    let r2_mag = norm(r2);
    let r3_mag = norm(r3);

    let z12 = cross(r1, r2);
    let z23 = cross(r2, r3);
    let z31 = cross(r3, r1);

    let n = [
        r1_mag * z23[0] + r2_mag * z31[0] + r3_mag * z12[0],
        r1_mag * z23[1] + r2_mag * z31[1] + r3_mag * z12[1],
        r1_mag * z23[2] + r2_mag * z31[2] + r3_mag * z12[2],
    ];
    let n_mag = norm(n);

    let d = [
        z12[0] + z23[0] + z31[0],
        z12[1] + z23[1] + z31[1],
        z12[2] + z23[2] + z31[2],
    ];
    let d_mag = norm(d);

    let s = [
        (r2_mag - r3_mag) * r1[0] + (r3_mag - r1_mag) * r2[0] + (r1_mag - r2_mag) * r3[0],
        (r2_mag - r3_mag) * r1[1] + (r3_mag - r1_mag) * r2[1] + (r1_mag - r2_mag) * r3[1],
        (r2_mag - r3_mag) * r1[2] + (r3_mag - r1_mag) * r2[2] + (r1_mag - r2_mag) * r3[2],
    ];
    let b = cross(d, r2);

    let lg = (mu / n_mag / d_mag).sqrt();
    [
        lg / r2_mag * b[0] + lg * s[0],
        lg / r2_mag * b[1] + lg * s[1],
        lg / r2_mag * b[2] + lg * s[2],
    ]
}

pub fn calc_herrick_gibbs_row(
    r1: [f64; 3],
    r2: [f64; 3],
    r3: [f64; 3],
    t1: f64,
    t2: f64,
    t3: f64,
    mu: f64,
) -> [f64; 3] {
    let t31 = t3 - t1;
    let t32 = t3 - t2;
    let t21 = t2 - t1;

    let r1_norm_cubed = norm(r1).powi(3);
    let r2_norm_cubed = norm(r2).powi(3);
    let r3_norm_cubed = norm(r3).powi(3);

    let c1 = -t32 * (1.0 / (t21 * t31) + mu / (12.0 * r1_norm_cubed));
    let c2 = (t32 - t21) * (1.0 / (t21 * t32) + mu / (12.0 * r2_norm_cubed));
    let c3 = t21 * (1.0 / (t32 * t31) + mu / (12.0 * r3_norm_cubed));

    [
        c1 * r1[0] + c2 * r2[0] + c3 * r3[0],
        c1 * r1[1] + c2 * r2[1] + c3 * r3[1],
        c1 * r1[2] + c2 * r2[2] + c3 * r3[2],
    ]
}

pub fn calc_gauss_row(
    r1: [f64; 3],
    r2: [f64; 3],
    r3: [f64; 3],
    t1: f64,
    t2: f64,
    t3: f64,
    mu: f64,
) -> [f64; 3] {
    let t12 = t1 - t2;
    let t32 = t3 - t2;
    let r2_mag_cubed = norm(r2).powi(3);

    let f1 = 1.0 - 0.5 * mu / r2_mag_cubed * t12.powi(2);
    let g1 = t12 - (1.0 / 6.0) * mu / r2_mag_cubed * t12.powi(2);
    let f3 = 1.0 - 0.5 * mu / r2_mag_cubed * t32.powi(2);
    let g3 = t32 - (1.0 / 6.0) * mu / r2_mag_cubed * t32.powi(2);

    let scale = 1.0 / (f1 * g3 - f3 * g1);
    [
        scale * (-f3 * r1[0] + f1 * r3[0]),
        scale * (-f3 * r1[1] + f1 * r3[1]),
        scale * (-f3 * r1[2] + f1 * r3[2]),
    ]
}

#[allow(clippy::too_many_arguments)]
pub fn gauss_iod_orbits_from_roots(
    r2_mags: &[f64],
    q1: [f64; 3],
    q2: [f64; 3],
    q3: [f64; 3],
    rho1_hat: [f64; 3],
    rho2_hat: [f64; 3],
    rho3_hat: [f64; 3],
    t1: f64,
    t2: f64,
    t3: f64,
    v: f64,
    velocity_method: i32,
    light_time: bool,
    mu: f64,
    c: f64,
) -> (Vec<f64>, Vec<f64>) {
    let mut epochs = Vec::with_capacity(r2_mags.len());
    let mut orbits_flat = Vec::with_capacity(r2_mags.len() * 6);

    let t31 = t3 - t1;
    let t21 = t2 - t1;
    let t32 = t3 - t2;

    for &r2_mag in r2_mags {
        let lambda1 = t32 / t31 * (1.0 + mu / (6.0 * r2_mag.powi(3)) * (t31.powi(2) - t32.powi(2)));
        let lambda3 = t21 / t31 * (1.0 + mu / (6.0 * r2_mag.powi(3)) * (t31.powi(2) - t21.powi(2)));

        let numerator = add(scale(q1, -lambda1), add(q2, scale(q3, -lambda3)));
        let rho1_mag = dot(numerator, cross(rho2_hat, rho3_hat)) / (lambda1 * v);
        let rho2_mag = dot(numerator, cross(rho1_hat, rho3_hat)) / v;
        let rho3_mag = dot(numerator, cross(rho1_hat, rho2_hat)) / (lambda3 * v);
        let rho1 = scale(rho1_hat, rho1_mag);
        let rho2 = scale(rho2_hat, rho2_mag);
        let rho3 = scale(rho3_hat, rho3_mag);

        if dot(rho2, rho2_hat) < 0.0 {
            continue;
        }

        let r1 = add(q1, rho1);
        let r2 = add(q2, rho2);
        let r3 = add(q3, rho3);

        let v2 = if velocity_method == 0 {
            calc_gauss_row(r1, r2, r3, t1, t2, t3, mu)
        } else if velocity_method == 1 {
            calc_gibbs_row(r1, r2, r3, mu)
        } else if velocity_method == 2 {
            calc_herrick_gibbs_row(r1, r2, r3, t1, t2, t3, mu)
        } else {
            continue;
        };

        let mut epoch = t2;
        if light_time {
            let lt = norm(sub(r2, q2)) / c;
            epoch -= lt;
        }

        let v_mag = norm(v2);
        if v_mag >= c {
            continue;
        }

        if norm(r2) > 300.0 || v_mag > 1.0 {
            continue;
        }

        if !epoch.is_finite()
            || !r2[0].is_finite()
            || !r2[1].is_finite()
            || !r2[2].is_finite()
            || !v2[0].is_finite()
            || !v2[1].is_finite()
            || !v2[2].is_finite()
        {
            continue;
        }

        epochs.push(epoch);
        orbits_flat.extend_from_slice(&[r2[0], r2[1], r2[2], v2[0], v2[1], v2[2]]);
    }

    (epochs, orbits_flat)
}
