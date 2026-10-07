//! Scientific assessment of orbit covariance and non-gravitational inputs.
//!
//! [`OrbitBatch::validate`](crate::OrbitBatch::validate) remains the structural,
//! fail-fast schema check.  This module assesses the numerical covariance and
//! non-gravitational semantics of every structurally valid orbit row without
//! mutating, repairing, or otherwise normalizing the supplied values.

use crate::{
    types::{SchemaError, SchemaResult},
    CoordinateRepresentation, CovarianceBatch, CovarianceUnits, NonGravitationalParametersRow,
    OrbitBatch,
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
    /// Canonical A1/A2/A3 coefficient values after representing null as zero.
    pub a_coefficients_with_null_as_zero: [f64; 3],
    /// Whether every canonical A1/A2/A3 coefficient is finite.
    pub a_coefficients_finite: bool,
    /// Whether at least one central A1/A2/A3 coefficient is nonzero, or `None`
    /// when non-finite coefficients prevent evaluation.
    pub nominal_a_coefficients_nonzero: Option<bool>,
    /// Whether the supplied covariance's semantic parameterization includes
    /// A1/A2/A3 in addition to Cartesian state. This does not assert that the
    /// covariance is numerically usable.
    pub covariance_includes_a1_a2_a3: bool,
    /// Whether the central A1/A2/A3 coefficients require a usable Marsden law,
    /// or `None` when non-finite coefficients prevent evaluation.
    pub nominal_requires_marsden_law: Option<bool>,
    /// Whether constructing uncertainty members from the supplied covariance
    /// parameterization requires a usable Marsden law.
    pub covariance_parameterization_requires_marsden_law: bool,
    /// Canonical shape of the supplied Marsden scalar-law fields.
    pub marsden_law_encoding: MarsdenLawEncoding,
    /// Whether the encoded Marsden law has scientifically valid values.
    pub marsden_law_values_valid: bool,
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
    /// The central coefficients or covariance parameterization require a law
    /// whose encoded values are invalid.
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
        let covariance = match orbits.coordinates.covariance.as_ref() {
            Some(covariance) => OrbitCovarianceAssessment::Present(
                assess_validated_orbit_covariance(covariance, row),
            ),
            None => OrbitCovarianceAssessment::Absent,
        };
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

/// Assess one supplied orbit-covariance row without modifying it.
///
/// The batch must use physical width six or nine, and `row` must address an
/// existing row. Padded coordinate-only rows in physical 9D storage are
/// assessed using their semantic 6D block.
pub fn assess_orbit_covariance(
    covariance: &CovarianceBatch,
    row: usize,
) -> SchemaResult<PresentOrbitCovarianceAssessment> {
    covariance.validate()?;
    if !matches!(covariance.dimension, 6 | 9) {
        return Err(SchemaError::InvalidCovarianceShape {
            rows: covariance.rows,
            dimension: covariance.dimension,
            values: covariance.values_row_major.len(),
        });
    }
    if row >= covariance.rows {
        return Err(SchemaError::InvalidRecordBatch(format!(
            "covariance row {row} is outside {} rows",
            covariance.rows
        )));
    }
    Ok(assess_validated_orbit_covariance(covariance, row))
}

fn assess_validated_orbit_covariance(
    covariance: &CovarianceBatch,
    row: usize,
) -> PresentOrbitCovarianceAssessment {
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

    PresentOrbitCovarianceAssessment {
        semantic_dimension,
        row_declared_valid,
        units_compatible,
        values_finite,
        positive_diagonal,
        sufficiently_symmetric,
        strictly_positive_definite,
        maximum_correlation_asymmetry,
        issues,
    }
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

/// Assess one non-gravitational parameter row without covariance support.
///
/// This is the canonical adam-core interpretation used by propagation
/// backends for a fixed supplied row.  Batch assessment additionally accounts
/// for semantic 9D covariance support through [`assess_orbit_batch`].
pub fn assess_non_gravitational_parameters(
    row: Option<NonGravitationalParametersRow>,
) -> NonGravitationalAssessment {
    assess_non_gravitational(row, None)
}

fn assess_non_gravitational(
    row: Option<NonGravitationalParametersRow>,
    covariance_dimension: Option<usize>,
) -> NonGravitationalAssessment {
    let a_coefficients_with_null_as_zero = row
        .map(|row| {
            [
                row.a1.unwrap_or(0.0),
                row.a2.unwrap_or(0.0),
                row.a3.unwrap_or(0.0),
            ]
        })
        .unwrap_or([0.0; 3]);
    let a_coefficients_finite = a_coefficients_with_null_as_zero
        .iter()
        .all(|value| value.is_finite());
    let nominal_a_coefficients_nonzero = a_coefficients_finite.then(|| {
        a_coefficients_with_null_as_zero
            .iter()
            .any(|value| *value != 0.0)
    });
    let covariance_includes_a1_a2_a3 = covariance_dimension == Some(9);
    let nominal_requires_marsden_law = nominal_a_coefficients_nonzero;
    let covariance_parameterization_requires_marsden_law = covariance_includes_a1_a2_a3;

    let constants = row
        .map(|row| [row.aln, row.nk, row.nm, row.nn, row.r0])
        .unwrap_or([None; 5]);
    let supplied_constant_count = constants.iter().filter(|value| value.is_some()).count();
    let marsden_law_encoding = match supplied_constant_count {
        0 => MarsdenLawEncoding::InverseSquare,
        5 => MarsdenLawEncoding::Complete,
        _ => MarsdenLawEncoding::Partial,
    };

    let mut issues = Vec::new();
    if !a_coefficients_finite {
        issues.push(NonGravitationalIssue::NonFiniteAccelerationCoefficient);
    }
    let marsden_law_values_valid = match marsden_law_encoding {
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
    let marsden_law_required = covariance_parameterization_requires_marsden_law
        || nominal_requires_marsden_law == Some(true);
    if marsden_law_required && !marsden_law_values_valid {
        issues.push(NonGravitationalIssue::RequiredMarsdenLawInvalid);
    }

    NonGravitationalAssessment {
        a_coefficients_with_null_as_zero,
        a_coefficients_finite,
        nominal_a_coefficients_nonzero,
        covariance_includes_a1_a2_a3,
        nominal_requires_marsden_law,
        covariance_parameterization_requires_marsden_law,
        marsden_law_encoding,
        marsden_law_values_valid,
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
    fn direct_covariance_assessment_matches_orbit_batch() {
        let covariance = cartesian_covariance(6, diagonal_covariance(6));
        let direct = assess_orbit_covariance(&covariance, 0).unwrap();
        let batch = assess_orbit_batch(&orbit(Some(covariance), None)).unwrap();
        assert_eq!(
            batch[0].covariance,
            OrbitCovarianceAssessment::Present(direct)
        );
    }

    #[test]
    fn direct_covariance_assessment_rejects_bad_address_and_dimension() {
        let covariance = cartesian_covariance(6, diagonal_covariance(6));
        assert!(matches!(
            assess_orbit_covariance(&covariance, 1),
            Err(SchemaError::InvalidRecordBatch(_))
        ));

        let unsupported = CovarianceBatch::new(
            1,
            7,
            diagonal_covariance(7),
            CovarianceUnits::Coordinate(CoordinateRepresentation::Cartesian),
        )
        .unwrap();
        assert!(matches!(
            assess_orbit_covariance(&unsupported, 0),
            Err(SchemaError::InvalidCovarianceShape { dimension: 7, .. })
        ));
    }

    #[test]
    fn absent_covariance_is_explicit() {
        let assessment = assess_orbit_batch(&orbit(None, None)).unwrap();
        assert_eq!(assessment[0].covariance, OrbitCovarianceAssessment::Absent);
        assert!(!assessment[0].non_gravitational.covariance_includes_a1_a2_a3);
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
        assert!(!assessment[0].non_gravitational.covariance_includes_a1_a2_a3);
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
        assert_eq!(non_grav.a_coefficients_with_null_as_zero, [0.0; 3]);
        assert_eq!(non_grav.nominal_a_coefficients_nonzero, Some(false));
        assert!(!non_grav.covariance_includes_a1_a2_a3);
        assert_eq!(non_grav.nominal_requires_marsden_law, Some(false));
        assert!(!non_grav.covariance_parameterization_requires_marsden_law);
        assert_eq!(
            non_grav.marsden_law_encoding,
            MarsdenLawEncoding::InverseSquare
        );
        assert!(non_grav.marsden_law_values_valid);
        assert!(non_grav.issues.is_empty());
    }

    #[test]
    fn nonzero_fixed_coefficients_with_inverse_square_and_custom_laws_are_valid() {
        let inverse_square = nongrav(Some(1.0e-9), None, None, None, None, None, None, None);
        let assessment = assess_orbit_batch(&orbit(None, Some(inverse_square))).unwrap();
        let non_grav = &assessment[0].non_gravitational;
        assert_eq!(non_grav.nominal_a_coefficients_nonzero, Some(true));
        assert_eq!(non_grav.nominal_requires_marsden_law, Some(true));
        assert!(!non_grav.covariance_parameterization_requires_marsden_law);
        assert_eq!(
            non_grav.marsden_law_encoding,
            MarsdenLawEncoding::InverseSquare
        );
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
        assert_eq!(non_grav.marsden_law_encoding, MarsdenLawEncoding::Complete);
        assert!(non_grav.marsden_law_values_valid);
        assert!(non_grav.issues.is_empty());
    }

    #[test]
    fn partial_marsden_law_is_reported_with_zero_central_coefficients() {
        let parameters = nongrav(None, None, None, Some(1.0), None, None, None, None);
        let assessment = assess_orbit_batch(&orbit(None, Some(parameters))).unwrap();
        let non_grav = &assessment[0].non_gravitational;
        assert_eq!(non_grav.nominal_requires_marsden_law, Some(false));
        assert!(!non_grav.covariance_parameterization_requires_marsden_law);
        assert_eq!(non_grav.marsden_law_encoding, MarsdenLawEncoding::Partial);
        assert_eq!(
            non_grav.issues,
            vec![NonGravitationalIssue::PartiallySpecifiedMarsdenLaw]
        );
    }

    #[test]
    fn nonzero_coefficients_with_partial_marsden_law_are_reported() {
        let parameters = nongrav(Some(1.0e-9), None, None, Some(1.0), None, None, None, None);
        let assessment = assess_orbit_batch(&orbit(None, Some(parameters))).unwrap();
        let non_grav = &assessment[0].non_gravitational;
        assert_eq!(non_grav.nominal_requires_marsden_law, Some(true));
        assert!(!non_grav.covariance_parameterization_requires_marsden_law);
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
        assert_eq!(non_grav.nominal_requires_marsden_law, Some(true));
        assert!(!non_grav.covariance_parameterization_requires_marsden_law);
        assert!(!non_grav.marsden_law_values_valid);
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
    fn nonfinite_a_coefficient_is_reported_without_gravity_fallback() {
        let parameters = nongrav(Some(f64::NAN), None, None, None, None, None, None, None);
        let assessment = assess_orbit_batch(&orbit(None, Some(parameters))).unwrap();
        let non_grav = &assessment[0].non_gravitational;
        assert!(!non_grav.a_coefficients_finite);
        assert_eq!(non_grav.nominal_a_coefficients_nonzero, None);
        assert_eq!(non_grav.nominal_requires_marsden_law, None);
        assert!(!non_grav.covariance_parameterization_requires_marsden_law);
        assert_eq!(
            non_grav.issues,
            vec![NonGravitationalIssue::NonFiniteAccelerationCoefficient]
        );
    }

    #[test]
    fn zero_central_coefficients_with_nine_dimensional_covariance_require_law() {
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
        assert_eq!(non_grav.nominal_a_coefficients_nonzero, Some(false));
        assert!(non_grav.covariance_includes_a1_a2_a3);
        assert_eq!(non_grav.nominal_requires_marsden_law, Some(false));
        assert!(non_grav.covariance_parameterization_requires_marsden_law);
        assert!(non_grav.marsden_law_values_valid);
        assert!(non_grav.issues.is_empty());
    }

    #[test]
    fn direct_nongrav_assessment_uses_adam_core_semantics() {
        let assessment = assess_non_gravitational_parameters(Some(NonGravitationalParametersRow {
            a1: Some(1.0e-9),
            a2: None,
            a3: None,
            aln: None,
            nk: None,
            nm: None,
            nn: None,
            r0: None,
        }));
        assert_eq!(
            assessment.a_coefficients_with_null_as_zero,
            [1.0e-9, 0.0, 0.0]
        );
        assert_eq!(assessment.nominal_a_coefficients_nonzero, Some(true));
        assert!(!assessment.covariance_includes_a1_a2_a3);
        assert_eq!(assessment.nominal_requires_marsden_law, Some(true));
        assert!(!assessment.covariance_parameterization_requires_marsden_law);
        assert_eq!(
            assessment.marsden_law_encoding,
            MarsdenLawEncoding::InverseSquare
        );
        assert!(assessment.marsden_law_values_valid);
        assert!(assessment.issues.is_empty());
    }

    #[test]
    fn absent_nongrav_with_nine_dimensional_covariance_uses_inverse_square_law() {
        let assessment = assess_orbit_batch(&orbit(
            Some(cartesian_covariance(9, diagonal_covariance(9))),
            None,
        ))
        .unwrap();
        let non_grav = &assessment[0].non_gravitational;
        assert_eq!(non_grav.a_coefficients_with_null_as_zero, [0.0; 3]);
        assert_eq!(non_grav.nominal_a_coefficients_nonzero, Some(false));
        assert!(non_grav.covariance_includes_a1_a2_a3);
        assert_eq!(non_grav.nominal_requires_marsden_law, Some(false));
        assert!(non_grav.covariance_parameterization_requires_marsden_law);
        assert_eq!(
            non_grav.marsden_law_encoding,
            MarsdenLawEncoding::InverseSquare
        );
        assert!(non_grav.marsden_law_values_valid);
        assert!(non_grav.issues.is_empty());
    }
}
