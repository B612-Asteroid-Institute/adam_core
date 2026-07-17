use super::request::EpochOrder;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropagationConvergenceStatus {
    Converged,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropagationFailureCode {
    NonFiniteInputState,
    NonFiniteOutputState,
    NonFiniteCovariance,
    SolverZeroDerivative,
    SolverMaxIterations,
    IntegratorFailure,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PropagationConvergence {
    pub output_row: usize,
    pub input_orbit_index: usize,
    pub input_time_index: usize,
    pub status: PropagationConvergenceStatus,
    pub backend: Option<String>,
    pub iterations: Option<usize>,
    pub failure_code: Option<PropagationFailureCode>,
    pub message: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PropagationDiagnostics {
    pub convergence: Vec<PropagationConvergence>,
    pub epoch_order: EpochOrder,
}

impl PropagationDiagnostics {
    pub fn failed_rows(&self) -> impl Iterator<Item = &PropagationConvergence> {
        self.convergence
            .iter()
            .filter(|row| row.status == PropagationConvergenceStatus::Failed)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RowOutput {
    pub states: Vec<[f64; 6]>,
    pub covariance: Option<Vec<[f64; 36]>>,
    pub covariance_validity: Option<Vec<bool>>,
    pub validity: Vec<bool>,
    pub messages: Vec<Option<String>>,
    pub backend: Option<String>,
    pub iterations: Vec<Option<usize>>,
    pub failure_codes: Vec<Option<PropagationFailureCode>>,
}

pub(super) fn failure_messages(codes: &[Option<PropagationFailureCode>]) -> Vec<Option<String>> {
    codes.iter().map(|code| code.map(failure_message)).collect()
}

pub(super) fn failure_message(code: PropagationFailureCode) -> String {
    match code {
        PropagationFailureCode::NonFiniteInputState => {
            "two-body propagation input state or gravitational parameter was non-finite".to_string()
        }
        PropagationFailureCode::NonFiniteOutputState => {
            "two-body propagation produced a non-finite state".to_string()
        }
        PropagationFailureCode::NonFiniteCovariance => {
            "two-body covariance propagation produced a non-finite covariance".to_string()
        }
        PropagationFailureCode::SolverZeroDerivative => {
            "two-body universal-anomaly solver encountered a zero derivative".to_string()
        }
        PropagationFailureCode::SolverMaxIterations => {
            "two-body universal-anomaly solver reached the maximum iteration count".to_string()
        }
        PropagationFailureCode::IntegratorFailure => {
            "propagator backend reported an integration failure".to_string()
        }
    }
}
