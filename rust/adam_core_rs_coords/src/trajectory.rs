//! Rust-owned validity-bounded trajectory operations.

use crate::{TimeArray, TimeScale};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrajectoryData {
    pub object_ids: Vec<String>,
    pub segment_ids: Vec<String>,
    pub coverage_start: TimeArray,
    pub coverage_end: TimeArray,
    pub orbit_times: TimeArray,
}

impl TrajectoryData {
    pub fn len(&self) -> usize {
        self.object_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.object_ids.is_empty()
    }

    pub fn validate_lengths(&self) -> Result<(), String> {
        let rows = self.len();
        for (field, actual) in [
            ("segment_id", self.segment_ids.len()),
            ("coverage_start", self.coverage_start.len()),
            ("coverage_end", self.coverage_end.len()),
            ("orbit.coordinates.time", self.orbit_times.len()),
        ] {
            if actual != rows {
                return Err(format!(
                    "trajectory column {field} has {actual} rows; expected {rows}"
                ));
            }
        }
        Ok(())
    }
}

pub fn trajectory_mjd(times: &TimeArray) -> Result<Vec<f64>, String> {
    times
        .rescale(TimeScale::Tdb)
        .map(|values| values.mjd_values())
        .map_err(|error| error.to_string())
}

pub fn trajectory_object_ids(data: &TrajectoryData) -> Vec<String> {
    let mut seen = HashSet::new();
    data.object_ids
        .iter()
        .filter(|object_id| seen.insert((*object_id).clone()))
        .cloned()
        .collect()
}

pub fn validate_trajectory(data: &TrajectoryData) -> Result<(), String> {
    data.validate_lengths()?;
    let starts = trajectory_mjd(&data.coverage_start)?;
    let ends = trajectory_mjd(&data.coverage_end)?;
    let epochs = trajectory_mjd(&data.orbit_times)?;
    if starts.iter().zip(&ends).any(|(start, end)| end <= start) {
        return Err("Trajectory coverage_end must be strictly after coverage_start".to_string());
    }
    if starts
        .iter()
        .zip(&ends)
        .zip(&epochs)
        .any(|((start, end), epoch)| epoch < start || epoch > end)
    {
        return Err(
            "Trajectory segment orbit epoch must lie within its coverage window".to_string(),
        );
    }

    let mut groups: HashMap<&str, Vec<usize>> = HashMap::new();
    let mut group_order = Vec::new();
    for (row, object_id) in data.object_ids.iter().enumerate() {
        if !groups.contains_key(object_id.as_str()) {
            group_order.push(object_id.as_str());
        }
        groups.entry(object_id).or_default().push(row);
    }
    for object_id in group_order {
        let rows = groups
            .get_mut(object_id)
            .expect("object group was inserted above");
        rows.sort_by(|&left, &right| starts[left].total_cmp(&starts[right]));
        for pair in rows.windows(2) {
            let left = pair[0];
            let right = pair[1];
            if starts[right] < ends[left] {
                return Err(format!(
                    "Trajectory has overlapping coverage windows for object {object_id:?}"
                ));
            }
        }
    }
    Ok(())
}

pub fn trajectory_segment_index(
    data: &TrajectoryData,
    time_mjd_tdb: f64,
    object_id: Option<&str>,
) -> Result<Option<i64>, String> {
    data.validate_lengths()?;
    if object_id.is_none() && trajectory_object_ids(data).len() > 1 {
        return Err(
            "segment_for requires object_id when the trajectory holds multiple objects".to_string(),
        );
    }
    let starts = trajectory_mjd(&data.coverage_start)?;
    let ends = trajectory_mjd(&data.coverage_end)?;
    let candidates: Vec<usize> = (0..data.len())
        .filter(|&row| {
            object_id.is_none_or(|value| data.object_ids[row] == value)
                && time_mjd_tdb >= starts[row]
                && time_mjd_tdb < ends[row]
        })
        .collect();
    match candidates.as_slice() {
        [] => Ok(None),
        [row] => Ok(Some(*row as i64)),
        _ => Err(format!(
            "Overlapping coverage windows match t={time_mjd_tdb}; trajectory is invalid"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> TrajectoryData {
        TrajectoryData {
            object_ids: vec!["a".to_string(), "a".to_string(), "b".to_string()],
            segment_ids: vec!["a0".to_string(), "a1".to_string(), "b0".to_string()],
            coverage_start: TimeArray::from_mjd(TimeScale::Tdb, &[1.0, 2.0, 1.0]).unwrap(),
            coverage_end: TimeArray::from_mjd(TimeScale::Tdb, &[2.0, 3.0, 3.0]).unwrap(),
            orbit_times: TimeArray::from_mjd(TimeScale::Tdb, &[1.5, 2.5, 2.0]).unwrap(),
        }
    }

    #[test]
    fn validates_touching_intervals_and_rejects_overlaps() {
        let mut data = fixture();
        assert!(validate_trajectory(&data).is_ok());
        data.coverage_start = TimeArray::from_mjd(TimeScale::Tdb, &[1.0, 1.5, 1.0]).unwrap();
        assert_eq!(
            validate_trajectory(&data).unwrap_err(),
            "Trajectory has overlapping coverage windows for object \"a\""
        );
    }

    #[test]
    fn selects_half_open_segment_for_explicit_object() {
        let data = fixture();
        assert_eq!(
            trajectory_segment_index(&data, 2.0, Some("a")).unwrap(),
            Some(1)
        );
        assert_eq!(
            trajectory_segment_index(&data, 3.0, Some("a")).unwrap(),
            None
        );
    }

    #[test]
    fn randomized_segment_lookup_matches_half_open_partition() {
        let rows = 256;
        let starts: Vec<f64> = (0..rows).map(|row| 100.0 + 2.0 * row as f64).collect();
        let ends: Vec<f64> = starts.iter().map(|start| start + 2.0).collect();
        let epochs: Vec<f64> = starts.iter().map(|start| start + 1.0).collect();
        let data = TrajectoryData {
            object_ids: vec!["random".to_string(); rows],
            segment_ids: (0..rows).map(|row| format!("segment-{row}")).collect(),
            coverage_start: TimeArray::from_mjd(TimeScale::Tdb, &starts).unwrap(),
            coverage_end: TimeArray::from_mjd(TimeScale::Tdb, &ends).unwrap(),
            orbit_times: TimeArray::from_mjd(TimeScale::Tdb, &epochs).unwrap(),
        };
        validate_trajectory(&data).unwrap();

        let mut state = 0x1234_5678_u64;
        for _ in 0..1_024 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let row = state as usize % rows;
            let fraction = ((state >> 16) % 1_000_000) as f64 / 1_000_000.0;
            let query = starts[row] + 2.0 * fraction;
            assert_eq!(
                trajectory_segment_index(&data, query, Some("random")).unwrap(),
                Some(row as i64)
            );
        }
        for (row, &end) in ends.iter().take(rows - 1).enumerate() {
            assert_eq!(
                trajectory_segment_index(&data, end, Some("random")).unwrap(),
                Some((row + 1) as i64)
            );
        }
    }
}
