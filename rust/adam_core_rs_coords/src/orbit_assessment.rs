//! Scientific assessment of orbit covariance and non-gravitational inputs.
//!
//! [`OrbitBatch::validate`](crate::OrbitBatch::validate) remains the structural,
//! fail-fast schema check.  This module assesses the numerical covariance and
//! non-gravitational semantics of every structurally valid orbit row without
//! mutating, repairing, or otherwise normalizing the supplied values.

use crate::{
    types::SchemaResult, CoordinateRepresentation, CovarianceBatch, CovarianceUnits,
    NonGravitationalParametersRow, OrbitBatch,
};

const MAX_COVARIANCE_DIMENSION: usize = 9;
const MAX_COVARIANCE_ELEMENTS: usize = MAX_COVARIANCE_DIMENSION * MAX_COVARIANCE_DIMENSION;

/// Maximum permitted absolute asymmetry after converting a covariance to
/// dimensionless correlation space.
pub const ORBIT_COVARIANCE_CORRELATION_SYMMETRY_TOLERANCE: f64 = 1.0e-10;

/// Assessment of every row in one structurally valid orbit batch.
#[derive(Clone, Debug, PartialEq)]
pub struct OrbitAssessment {
    /// Input row addressed by this assessment.
    pub row: usize,
    /// Supplied covariance assessment.
    pub covariance: OrbitCovarianceAssessment,
    /// Supplied non-gravitational parameter assessment.
    pub non_gravitational: NonGravitationalAssessment,
}

/// Assessment of the covariance attached to one orbit row.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum OrbitCovarianceAssessment {
    /// The orbit row has no supplied covariance.
    Absent,
    /// The orbit row has a supplied covariance.
    Present(PresentOrbitCovarianceAssessment),
}

impl OrbitCovarianceAssessment {
    fn semantic_dimension(&self) -> Option<usize> {
        match self {
            Self::Absent => None,
            Self::Present(assessment) => Some(assessment.semantic_dimension),
        }
    }
}

/// Numerical checks for one present covariance row.
#[derive(Clone, Debug, PartialEq)]
pub struct PresentOrbitCovarianceAssessment {
    /// Semantic solved dimension after recognizing canonical padded 6D rows in
    /// physical 9D storage.
    pub semantic_dimension: usize,
    /// Whether the covariance validity bitmap marks this row valid.
    pub row_declared_valid: bool,
    /// Whether covariance units match Cartesian orbit coordinates.
    pub units_compatible: bool,
    /// Whether all consumed covariance and scaled-correlation values are finite.
    pub values_finite: bool,
    /// Whether every consumed diagonal variance is strictly positive.
    pub positive_diagonal: bool,
    /// Whether correlation-space asymmetry is within the strict tolerance, or
    /// `None` when prerequisite numerical checks prevented evaluation.
    pub sufficiently_symmetric: Option<bool>,
    /// Whether strict scaled Cholesky factorization succeeds unchanged, or
    /// `None` when symmetry or an earlier prerequisite prevented evaluation.
    pub strictly_positive_definite: Option<bool>,
    /// Maximum absolute correlation-space asymmetry, when computable.
    pub maximum_correlation_asymmetry: Option<f64>,
    /// Independent issues observed for this row, in deterministic check order.
    pub issues: Vec<OrbitCovarianceIssue>,
}

/// Typed covariance issue reported without modifying the supplied values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum OrbitCovarianceIssue {
    /// The covariance validity bitmap marks the row invalid.
    InvalidRow,
    /// Covariance units do not describe Cartesian coordinate covariance.
    UnsupportedUnits,
    /// A consumed covariance or scaled-correlation value is non-finite.
    NonFiniteValue,
    /// A consumed diagonal variance is zero or negative.
    NonPositiveDiagonal,
    /// Correlation-space asymmetry exceeds the strict tolerance.
    ExcessiveAsymmetry,
    /// Strict scaled Cholesky factorization failed.
    NotPositiveDefinite,
}

/// Canonical interpretation of the five Marsden scalar-law fields.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum MarsdenLawEncoding {
    /// All constants are absent, selecting the canonical inverse-square law.
    InverseSquare,
    /// All ALN/NK/NM/NN/R0 constants are present.
    Complete,
    /// Some, but not all, Marsden constants are present.
    Partial,
}

/// Assessment of one orbit row's non-gravitational parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct NonGravitationalAssessment {
    /// Effective A1/A2/A3 values; null inputs are represented as zero.
    pub effective_acceleration: [f64; 3],
    /// Whether every effective A coefficient is finite.
    pub coefficients_finite: bool,
    /// Whether at least one nominal A coefficient is nonzero, or `None` when
    /// non-finite coefficients make nominal activity indeterminate.
    pub nominal_active: Option<bool>,
    /// Whether a semantic 9D covariance can produce nonzero A members.
    pub covariance_may_activate: bool,
    /// Whether nominal or covariance support requires a usable force law, or
    /// `None` when nominal activity is indeterminate and covariance does not
    /// independently require the law.
    pub law_required: Option<bool>,
    /// Canonical shape of the supplied Marsden scalar-law fields.
    pub law_encoding: MarsdenLawEncoding,
    /// Whether the encoded law has scientifically valid numerical values.
    pub law_values_valid: bool,
    /// Independent issues observed for this row, in deterministic check order.
    pub issues: Vec<NonGravitationalIssue>,
}

/// Typed non-gravitational issue reported without changing supplied metadata.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum NonGravitationalIssue {
    /// At least one present A1/A2/A3 value is non-finite.
    NonFiniteAccelerationCoefficient,
    /// Some, but not all, Marsden scalar-law constants are present.
    PartiallySpecifiedMarsdenLaw,
    /// At least one complete Marsden scalar-law constant is non-finite.
    NonFiniteMarsdenConstant,
    /// ALN is finite but not strictly positive.
    NonPositiveMarsdenAlpha,
    /// R0 is finite but not strictly positive.
    NonPositiveMarsdenScale,
    /// The represented nominal/covariance support needs a law that is invalid.
    RequiredMarsdenLawInvalid,
}

/// Assess covariance and non-gravitational semantics for every orbit row.
///
/// Structural validation runs once before rows are addressed.  The returned
/// vector is in input row order and always has `orbits.len()` entries.
pub fn assess_orbit_batch(orbits: &OrbitBatch) -> SchemaResult<Vec<OrbitAssessment>> {
    orbits.validate()?;
    let mut assessments = Vec::with_capacity(orbits.len());
    for row in 0..orbits.len() {
        let covariance = assess_covariance(orbits.coordinates.covariance.as_ref(), row);
        let non_gravitational = assess_non_gravitational(
            orbits
                .non_gravitational_parameters
                .as_ref()
                .map(|parameters| parameters.row(row)),
            covariance.semantic_dimension(),
        );
        assessments.push(OrbitAssessment {
            row,
            covariance,
            non_gravitational,
        });
    }
    Ok(assessments)
}

fn assess_covariance(
    covariance: Option<&CovarianceBatch>,
    row: usize,
) -> OrbitCovarianceAssessment {
    let Some(covariance) = covariance else {
        return OrbitCovarianceAssessment::Absent;
    };
    let semantic_dimension = covariance.row_dimension(row);
    let row_declared_valid = covariance.is_row_valid(row);
    let units_compatible = matches!(
        &covariance.units,
        CovarianceUnits::Coordinate(CoordinateRepresentation::Cartesian)
    );
    let mut issues = Vec::new();
    if !row_declared_valid {
        issues.push(OrbitCovarianceIssue::InvalidRow);
    }
    if !units_compatible {
        issues.push(OrbitCovarianceIssue::UnsupportedUnits);
    }

    let values = semantic_covariance_row(covariance, row, semantic_dimension);
    let raw_values_finite = values[..semantic_dimension * semantic_dimension]
        .iter()
        .all(|value| value.is_finite());
    if !raw_values_finite {
        issues.push(OrbitCovarianceIssue::NonFiniteValue);
    }

    let positive_diagonal = (0..semantic_dimension).all(|axis| {
        let value = values[axis * semantic_dimension + axis];
        value.is_finite() && value > 0.0
    });
    if raw_values_finite && !positive_diagonal {
        issues.push(OrbitCovarianceIssue::NonPositiveDiagonal);
    }

    let mut values_finite = raw_values_finite;
    let mut sufficiently_symmetric = None;
    let mut strictly_positive_definite = None;
    let mut maximum_correlation_asymmetry = None;
    if raw_values_finite && positive_diagonal {
        let mut scales = [0.0; MAX_COVARIANCE_DIMENSION];
        for axis in 0..semantic_dimension {
            scales[axis] = values[axis * semantic_dimension + axis].sqrt();
        }
        let mut correlation = [0.0; MAX_COVARIANCE_ELEMENTS];
        for r in 0..semantic_dimension {
            for c in 0..semantic_dimension {
                correlation[r * semantic_dimension + c] =
                    values[r * semantic_dimension + c] / (scales[r] * scales[c]);
            }
        }
        if correlation[..semantic_dimension * semantic_dimension]
            .iter()
            .all(|value| value.is_finite())
        {
            let maximum_asymmetry = maximum_asymmetry(&correlation, semantic_dimension);
            maximum_correlation_asymmetry = Some(maximum_asymmetry);
            let symmetric = maximum_asymmetry <= ORBIT_COVARIANCE_CORRELATION_SYMMETRY_TOLERANCE;
            sufficiently_symmetric = Some(symmetric);
            if symmetric {
                let positive_definite = strict_cholesky(&correlation, semantic_dimension).is_ok();
                strictly_positive_definite = Some(positive_definite);
                if !positive_definite {
                    issues.push(OrbitCovarianceIssue::NotPositiveDefinite);
                }
            } else {
                issues.push(OrbitCovarianceIssue::ExcessiveAsymmetry);
            }
        } else {
            values_finite = false;
            push_unique(&mut issues, OrbitCovarianceIssue::NonFiniteValue);
        }
    }

    OrbitCovarianceAssessment::Present(PresentOrbitCovarianceAssessment {
        semantic_dimension,
        row_declared_valid,
        units_compatible,
        values_finite,
        positive_diagonal,
        sufficiently_symmetric,
        strictly_positive_definite,
        maximum_correlation_asymmetry,
        issues,
    })
}

fn semantic_covariance_row(
    covariance: &CovarianceBatch,
    row: usize,
    semantic_dimension: usize,
) -> [f64; MAX_COVARIANCE_ELEMENTS] {
    let mut values = [0.0; MAX_COVARIANCE_ELEMENTS];
    let source = covariance.row_values(row);
    for r in 0..semantic_dimension {
        for c in 0..semantic_dimension {
            values[r * semantic_dimension + c] = source[r * covariance.dimension + c];
        }
    }
    values
}

fn maximum_asymmetry(values: &[f64; MAX_COVARIANCE_ELEMENTS], dimension: usize) -> f64 {
    let mut maximum = 0.0_f64;
    for r in 0..dimension {
        for c in (r + 1)..dimension {
            maximum = maximum.max((values[r * dimension + c] - values[c * dimension + r]).abs());
        }
    }
    maximum
}

fn strict_cholesky(values: &[f64; MAX_COVARIANCE_ELEMENTS], dimension: usize) -> Result<(), ()> {
    let mut lower = [0.0; MAX_COVARIANCE_ELEMENTS];
    for row in 0..dimension {
        for column in 0..=row {
            let mut value = values[row * dimension + column];
            for inner in 0..column {
                value -= lower[row * dimension + inner] * lower[column * dimension + inner];
            }
            if row == column {
                if !value.is_finite() || value <= 0.0 {
                    return Err(());
                }
                lower[row * dimension + column] = value.sqrt();
            } else {
                lower[row * dimension + column] = value / lower[column * dimension + column];
            }
        }
    }
    Ok(())
}

fn assess_non_gravitational(
    row: Option<NonGravitationalParametersRow>,
    covariance_dimension: Option<usize>,
) -> NonGravitationalAssessment {
    let effective_acceleration = row
        .map(|row| {
            [
                row.a1.unwrap_or(0.0),
                row.a2.unwrap_or(0.0),
                row.a3.unwrap_or(0.0),
            ]
        })
        .unwrap_or([0.0; 3]);
    let coefficients_finite = effective_acceleration.iter().all(|value| value.is_finite());
    let nominal_active =
        coefficients_finite.then(|| effective_acceleration.iter().any(|value| *value != 0.0));
    let covariance_may_activate = covariance_dimension == Some(9);
    let law_required = if covariance_may_activate {
        Some(true)
    } else {
        nominal_active
    };

    let constants = row
        .map(|row| [row.aln, row.nk, row.nm, row.nn, row.r0])
        .unwrap_or([None; 5]);
    let supplied_constant_count = constants.iter().filter(|value| value.is_some()).count();
    let law_encoding = match supplied_constant_count {
        0 => MarsdenLawEncoding::InverseSquare,
        5 => MarsdenLawEncoding::Complete,
        _ => MarsdenLawEncoding::Partial,
    };

    let mut issues = Vec::new();
    if !coefficients_finite {
        issues.push(NonGravitationalIssue::NonFiniteAccelerationCoefficient);
    }
    let law_values_valid = match law_encoding {
        MarsdenLawEncoding::InverseSquare => true,
        MarsdenLawEncoding::Partial => {
            issues.push(NonGravitationalIssue::PartiallySpecifiedMarsdenLaw);
            false
        }
        MarsdenLawEncoding::Complete => {
            let values = constants.map(|value| value.expect("complete Marsden constants"));
            let finite = values.iter().all(|value| value.is_finite());
            if !finite {
                issues.push(NonGravitationalIssue::NonFiniteMarsdenConstant);
            }
            let alpha_positive = values[0].is_finite() && values[0] > 0.0;
            if values[0].is_finite() && !alpha_positive {
                issues.push(NonGravitationalIssue::NonPositiveMarsdenAlpha);
            }
            let scale_positive = values[4].is_finite() && values[4] > 0.0;
            if values[4].is_finite() && !scale_positive {
                issues.push(NonGravitationalIssue::NonPositiveMarsdenScale);
            }
            finite && alpha_positive && scale_positive
        }
    };
    if law_required == Some(true) && !law_values_valid {
        issues.push(NonGravitationalIssue::RequiredMarsdenLawInvalid);
    }

    NonGravitationalAssessment {
        effective_acceleration,
        coefficients_finite,
        nominal_active,
        covariance_may_activate,
        law_required,
        law_encoding,
        law_values_valid,
        issues,
    }
}

fn push_unique<T: Copy + PartialEq>(values: &mut Vec<T>, value: T) {
    if !values.contains(&value) {
        values.push(value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CoordinateBatch, CovarianceBatch, Epoch, NonGravitationalParametersBatch, OrbitId,
        OriginArray, OriginId, TimeArray, TimeScale, Validity,
    };

    fn diagonal_covariance(dimension: usize) -> Vec<f64> {
        let mut values = vec![0.0; dimension * dimension];
        for axis in 0..dimension {
            values[axis * dimension + axis] = (axis + 1) as f64;
        }
        values
    }

    fn padded_six_in_nine_covariance() -> Vec<f64> {
        let mut values = vec![f64::NAN; 81];
        for r in 0..6 {
            for c in 0..6 {
                values[r * 9 + c] = if r == c { (r + 1) as f64 } else { 0.0 };
            }
        }
        values
    }

    fn orbit(
        covariance: Option<CovarianceBatch>,
        non_gravitational_parameters: Option<NonGravitationalParametersBatch>,
    ) -> OrbitBatch {
        let coordinates = CoordinateBatch::cartesian(
            vec![[1.0, 0.0, 0.0, 0.0, 0.01, 0.0]],
            crate::DataFrame::Ecliptic,
            OriginArray::repeat(OriginId::Naif(10), 1),
            Some(TimeArray::new(TimeScale::Tdb, vec![Epoch::new(60_000, 0)]).unwrap()),
            covariance,
        )
        .unwrap();
        let orbit =
            OrbitBatch::new(vec![OrbitId("test".to_string())], vec![None], coordinates).unwrap();
        match non_gravitational_parameters {
            Some(parameters) => orbit.with_non_gravitational_parameters(parameters).unwrap(),
            None => orbit,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn nongrav(
        a1: Option<f64>,
        a2: Option<f64>,
        a3: Option<f64>,
        aln: Option<f64>,
        nk: Option<f64>,
        nm: Option<f64>,
        nn: Option<f64>,
        r0: Option<f64>,
    ) -> NonGravitationalParametersBatch {
        NonGravitationalParametersBatch {
            source: vec![Some("test".to_string())],
            a1: vec![a1],
            a2: vec![a2],
            a3: vec![a3],
            aln: vec![aln],
            nk: vec![nk],
            nm: vec![nm],
            nn: vec![nn],
            r0: vec![r0],
        }
    }

    fn cartesian_covariance(dimension: usize, values: Vec<f64>) -> CovarianceBatch {
        CovarianceBatch::new(
            1,
            dimension,
            values,
            CovarianceUnits::Coordinate(CoordinateRepresentation::Cartesian),
        )
        .unwrap()
    }

    fn present(assessment: &OrbitAssessment) -> &PresentOrbitCovarianceAssessment {
        match &assessment.covariance {
            OrbitCovarianceAssessment::Present(covariance) => covariance,
            OrbitCovarianceAssessment::Absent => panic!("expected covariance"),
        }
    }

    #[test]
    fn absent_covariance_is_explicit() {
        let assessment = assess_orbit_batch(&orbit(None, None)).unwrap();
        assert_eq!(assessment[0].covariance, OrbitCovarianceAssessment::Absent);
        assert!(!assessment[0].non_gravitational.covariance_may_activate);
    }

    #[test]
    fn strict_valid_six_and_nine_dimensional_covariance_pass() {
        for dimension in [6, 9] {
            let assessment = assess_orbit_batch(&orbit(
                Some(cartesian_covariance(
                    dimension,
                    diagonal_covariance(dimension),
                )),
                None,
            ))
            .unwrap();
            let covariance = present(&assessment[0]);
            assert_eq!(covariance.semantic_dimension, dimension);
            assert!(covariance.values_finite);
            assert!(covariance.positive_diagonal);
            assert_eq!(covariance.sufficiently_symmetric, Some(true));
            assert_eq!(covariance.strictly_positive_definite, Some(true));
            assert_eq!(covariance.maximum_correlation_asymmetry, Some(0.0));
            assert!(covariance.issues.is_empty());
        }
    }

    #[test]
    fn padded_six_dimensional_covariance_uses_only_semantic_block() {
        let assessment = assess_orbit_batch(&orbit(
            Some(cartesian_covariance(9, padded_six_in_nine_covariance())),
            None,
        ))
        .unwrap();
        let covariance = present(&assessment[0]);
        assert_eq!(covariance.semantic_dimension, 6);
        assert_eq!(covariance.strictly_positive_definite, Some(true));
        assert!(!assessment[0].non_gravitational.covariance_may_activate);
    }

    #[test]
    fn covariance_metadata_issues_do_not_hide_numerical_assessment() {
        let invalid = cartesian_covariance(6, diagonal_covariance(6))
            .with_row_validity(Validity::from_bools(&[false]))
            .unwrap();
        let assessment = assess_orbit_batch(&orbit(Some(invalid), None)).unwrap();
        let covariance = present(&assessment[0]);
        assert_eq!(covariance.strictly_positive_definite, Some(true));
        assert_eq!(covariance.issues, vec![OrbitCovarianceIssue::InvalidRow]);

        let unsupported = CovarianceBatch::new(
            1,
            6,
            diagonal_covariance(6),
            CovarianceUnits::ObservationAngular2D,
        )
        .unwrap();
        let assessment = assess_orbit_batch(&orbit(Some(unsupported), None)).unwrap();
        let covariance = present(&assessment[0]);
        assert_eq!(covariance.strictly_positive_definite, Some(true));
        assert_eq!(
            covariance.issues,
            vec![OrbitCovarianceIssue::UnsupportedUnits]
        );
    }

    #[test]
    fn covariance_numerical_failures_are_distinguished() {
        let mut nonfinite = diagonal_covariance(6);
        nonfinite[1] = f64::INFINITY;
        let assessment =
            assess_orbit_batch(&orbit(Some(cartesian_covariance(6, nonfinite)), None)).unwrap();
        let covariance = present(&assessment[0]);
        assert_eq!(covariance.sufficiently_symmetric, None);
        assert_eq!(covariance.strictly_positive_definite, None);
        assert_eq!(
            covariance.issues,
            vec![OrbitCovarianceIssue::NonFiniteValue]
        );

        let mut nonpositive = diagonal_covariance(6);
        nonpositive[0] = 0.0;
        let assessment =
            assess_orbit_batch(&orbit(Some(cartesian_covariance(6, nonpositive)), None)).unwrap();
        let covariance = present(&assessment[0]);
        assert_eq!(covariance.sufficiently_symmetric, None);
        assert_eq!(covariance.strictly_positive_definite, None);
        assert_eq!(
            covariance.issues,
            vec![OrbitCovarianceIssue::NonPositiveDiagonal]
        );

        let mut asymmetric = diagonal_covariance(6);
        asymmetric[1] = 2.0e-10;
        let assessment =
            assess_orbit_batch(&orbit(Some(cartesian_covariance(6, asymmetric)), None)).unwrap();
        let covariance = present(&assessment[0]);
        assert_eq!(covariance.sufficiently_symmetric, Some(false));
        assert_eq!(covariance.strictly_positive_definite, None);
        assert_eq!(
            covariance.issues,
            vec![OrbitCovarianceIssue::ExcessiveAsymmetry]
        );

        let mut indefinite = diagonal_covariance(6);
        indefinite[1] = 2.0;
        indefinite[6] = 2.0;
        let assessment =
            assess_orbit_batch(&orbit(Some(cartesian_covariance(6, indefinite)), None)).unwrap();
        let covariance = present(&assessment[0]);
        assert_eq!(covariance.sufficiently_symmetric, Some(true));
        assert_eq!(covariance.strictly_positive_definite, Some(false));
        assert_eq!(
            covariance.issues,
            vec![OrbitCovarianceIssue::NotPositiveDefinite]
        );
    }

    #[test]
    fn all_null_nongrav_is_gravity_only_even_with_source_provenance() {
        let parameters = nongrav(None, None, None, None, None, None, None, None);
        let assessment = assess_orbit_batch(&orbit(None, Some(parameters))).unwrap();
        let non_grav = &assessment[0].non_gravitational;
        assert_eq!(non_grav.effective_acceleration, [0.0; 3]);
        assert_eq!(non_grav.nominal_active, Some(false));
        assert!(!non_grav.covariance_may_activate);
        assert_eq!(non_grav.law_required, Some(false));
        assert_eq!(non_grav.law_encoding, MarsdenLawEncoding::InverseSquare);
        assert!(non_grav.law_values_valid);
        assert!(non_grav.issues.is_empty());
    }

    #[test]
    fn fixed_active_inverse_square_and_complete_custom_laws_are_valid() {
        let inverse_square = nongrav(Some(1.0e-9), None, None, None, None, None, None, None);
        let assessment = assess_orbit_batch(&orbit(None, Some(inverse_square))).unwrap();
        let non_grav = &assessment[0].non_gravitational;
        assert_eq!(non_grav.nominal_active, Some(true));
        assert_eq!(non_grav.law_required, Some(true));
        assert_eq!(non_grav.law_encoding, MarsdenLawEncoding::InverseSquare);
        assert!(non_grav.issues.is_empty());

        let custom = nongrav(
            Some(1.0e-9),
            Some(-2.0e-10),
            None,
            Some(1.0),
            Some(0.0),
            Some(2.0),
            Some(5.0),
            Some(1.0),
        );
        let assessment = assess_orbit_batch(&orbit(None, Some(custom))).unwrap();
        let non_grav = &assessment[0].non_gravitational;
        assert_eq!(non_grav.law_encoding, MarsdenLawEncoding::Complete);
        assert!(non_grav.law_values_valid);
        assert!(non_grav.issues.is_empty());
    }

    #[test]
    fn partial_marsden_law_is_reported_even_when_inactive() {
        let parameters = nongrav(None, None, None, Some(1.0), None, None, None, None);
        let assessment = assess_orbit_batch(&orbit(None, Some(parameters))).unwrap();
        let non_grav = &assessment[0].non_gravitational;
        assert_eq!(non_grav.law_required, Some(false));
        assert_eq!(non_grav.law_encoding, MarsdenLawEncoding::Partial);
        assert_eq!(
            non_grav.issues,
            vec![NonGravitationalIssue::PartiallySpecifiedMarsdenLaw]
        );
    }

    #[test]
    fn active_partial_marsden_law_is_required_and_invalid() {
        let parameters = nongrav(Some(1.0e-9), None, None, Some(1.0), None, None, None, None);
        let assessment = assess_orbit_batch(&orbit(None, Some(parameters))).unwrap();
        let non_grav = &assessment[0].non_gravitational;
        assert_eq!(non_grav.law_required, Some(true));
        assert_eq!(
            non_grav.issues,
            vec![
                NonGravitationalIssue::PartiallySpecifiedMarsdenLaw,
                NonGravitationalIssue::RequiredMarsdenLawInvalid,
            ]
        );
    }

    #[test]
    fn required_invalid_marsden_law_reports_atomic_and_combination_issues() {
        let parameters = nongrav(
            Some(1.0e-9),
            None,
            None,
            Some(-1.0),
            Some(4.0),
            Some(2.0),
            Some(5.0),
            Some(f64::INFINITY),
        );
        let assessment = assess_orbit_batch(&orbit(None, Some(parameters))).unwrap();
        let non_grav = &assessment[0].non_gravitational;
        assert_eq!(non_grav.law_required, Some(true));
        assert!(!non_grav.law_values_valid);
        assert_eq!(
            non_grav.issues,
            vec![
                NonGravitationalIssue::NonFiniteMarsdenConstant,
                NonGravitationalIssue::NonPositiveMarsdenAlpha,
                NonGravitationalIssue::RequiredMarsdenLawInvalid,
            ]
        );
    }

    #[test]
    fn nonfinite_acceleration_is_reported_without_gravity_fallback() {
        let parameters = nongrav(Some(f64::NAN), None, None, None, None, None, None, None);
        let assessment = assess_orbit_batch(&orbit(None, Some(parameters))).unwrap();
        let non_grav = &assessment[0].non_gravitational;
        assert!(!non_grav.coefficients_finite);
        assert_eq!(non_grav.nominal_active, None);
        assert_eq!(non_grav.law_required, None);
        assert_eq!(
            non_grav.issues,
            vec![NonGravitationalIssue::NonFiniteAccelerationCoefficient]
        );
    }

    #[test]
    fn zero_nominal_with_nine_dimensional_covariance_requires_force_law() {
        let parameters = nongrav(
            Some(0.0),
            Some(0.0),
            None,
            Some(0.111_262_042_6),
            Some(4.6142),
            Some(2.15),
            Some(5.093),
            Some(2.808),
        );
        let assessment = assess_orbit_batch(&orbit(
            Some(cartesian_covariance(9, diagonal_covariance(9))),
            Some(parameters),
        ))
        .unwrap();
        let non_grav = &assessment[0].non_gravitational;
        assert_eq!(non_grav.nominal_active, Some(false));
        assert!(non_grav.covariance_may_activate);
        assert_eq!(non_grav.law_required, Some(true));
        assert!(non_grav.law_values_valid);
        assert!(non_grav.issues.is_empty());
    }

    #[test]
    fn absent_nongrav_with_nine_dimensional_covariance_uses_inverse_square_law() {
        let assessment = assess_orbit_batch(&orbit(
            Some(cartesian_covariance(9, diagonal_covariance(9))),
            None,
        ))
        .unwrap();
        let non_grav = &assessment[0].non_gravitational;
        assert_eq!(non_grav.effective_acceleration, [0.0; 3]);
        assert!(non_grav.covariance_may_activate);
        assert_eq!(non_grav.law_required, Some(true));
        assert_eq!(non_grav.law_encoding, MarsdenLawEncoding::InverseSquare);
        assert!(non_grav.law_values_valid);
        assert!(non_grav.issues.is_empty());
    }
}
