//! Canonical batch membership planning for orbit determination.
//!
//! The planners are intentionally science-free: they parse already decoded
//! canonical schemas, validate every association atomically, and materialize
//! deterministic group/member row plans before a propagation provider or an
//! observation pipeline is touched.

use crate::observations::AssociationBatch;
use crate::{OrbitBatch, OrbitId};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssociationGroup {
    pub group_id: String,
    /// Source-catalog rows, always in source-table order.
    pub source_indices: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrbitMemberGroup {
    pub orbit_id: String,
    pub orbit_index: usize,
    /// Observation rows, always in source-table order.
    pub source_indices: Vec<usize>,
    /// Rows in the caller's FittedOrbitMembers table corresponding one-to-one
    /// with `source_indices`.
    pub member_indices: Vec<usize>,
}

fn unique_source_rows(source_ids: &[String]) -> Result<HashMap<&str, usize>, String> {
    let mut rows = HashMap::with_capacity(source_ids.len());
    for (row, id) in source_ids.iter().enumerate() {
        if id.trim().is_empty() {
            return Err(format!("observation ID at row {row} is empty"));
        }
        if rows.insert(id.as_str(), row).is_some() {
            return Err(format!("observation ID {id:?} is not unique"));
        }
    }
    Ok(rows)
}

/// Validate raw IOD/OD associations and produce first-appearance group order
/// with source-table member order. A detection may appear in multiple groups,
/// but the same `(object_id, detection_id)` pair may appear only once.
pub fn plan_associations(
    source_ids: &[String],
    associations: &AssociationBatch,
) -> Result<Vec<AssociationGroup>, String> {
    associations.validate().map_err(|error| error.to_string())?;
    let source_rows = unique_source_rows(source_ids)?;
    let mut order = Vec::<String>::new();
    let mut members = HashMap::<String, HashSet<usize>>::new();
    let mut pairs = HashSet::<(String, usize)>::new();

    for row in 0..associations.len() {
        let detection_id = &associations.detection_id[row];
        let source_index = source_rows
            .get(detection_id.as_str())
            .copied()
            .ok_or_else(|| {
                format!("association row {row} references unknown detection_id {detection_id:?}")
            })?;
        let group_id = associations.object_id[row]
            .as_ref()
            .ok_or_else(|| format!("association row {row} has a null object_id"))?;
        if group_id.trim().is_empty() {
            return Err(format!("association row {row} has an empty object_id"));
        }
        if !members.contains_key(group_id) {
            order.push(group_id.clone());
            members.insert(group_id.clone(), HashSet::new());
        }
        if !pairs.insert((group_id.clone(), source_index)) {
            return Err(format!(
                "duplicate association pair ({group_id:?}, {detection_id:?})"
            ));
        }
        members
            .get_mut(group_id)
            .expect("group inserted above")
            .insert(source_index);
    }

    Ok(order
        .into_iter()
        .map(|group_id| {
            let selected = &members[&group_id];
            AssociationGroup {
                group_id,
                source_indices: (0..source_ids.len())
                    .filter(|row| selected.contains(row))
                    .collect(),
            }
        })
        .collect())
}

/// Validate explicit fitted-orbit membership. Group order is the first
/// appearance of each orbit ID in the members table; rows inside a group are
/// always restored to source-observation order.
pub fn plan_orbit_members(
    orbits: &OrbitBatch,
    source_ids: &[String],
    member_orbit_ids: &[String],
    member_obs_ids: &[String],
) -> Result<Vec<OrbitMemberGroup>, String> {
    orbits.validate().map_err(|error| error.to_string())?;
    if member_orbit_ids.len() != member_obs_ids.len() {
        return Err(format!(
            "FittedOrbitMembers orbit_id/obs_id length mismatch: {} != {}",
            member_orbit_ids.len(),
            member_obs_ids.len()
        ));
    }
    let source_rows = unique_source_rows(source_ids)?;
    let mut orbit_rows = HashMap::<&str, usize>::with_capacity(orbits.len());
    for (row, OrbitId(id)) in orbits.orbit_id.iter().enumerate() {
        if id.trim().is_empty() {
            return Err(format!("orbit ID at row {row} is empty"));
        }
        if orbit_rows.insert(id.as_str(), row).is_some() {
            return Err(format!("orbit ID {id:?} is not unique"));
        }
    }

    let mut order = Vec::<String>::new();
    let mut by_orbit = HashMap::<String, HashMap<usize, usize>>::new();
    let mut pairs = HashSet::<(String, usize)>::new();
    for row in 0..member_orbit_ids.len() {
        let orbit_id = &member_orbit_ids[row];
        if !orbit_rows.contains_key(orbit_id.as_str()) {
            return Err(format!(
                "FittedOrbitMembers row {row} references unknown orbit_id {orbit_id:?}"
            ));
        }
        let obs_id = &member_obs_ids[row];
        let source_index = source_rows.get(obs_id.as_str()).copied().ok_or_else(|| {
            format!("FittedOrbitMembers row {row} references unknown obs_id {obs_id:?}")
        })?;
        if !by_orbit.contains_key(orbit_id) {
            order.push(orbit_id.clone());
            by_orbit.insert(orbit_id.clone(), HashMap::new());
        }
        if !pairs.insert((orbit_id.clone(), source_index)) {
            return Err(format!(
                "duplicate fitted-orbit member pair ({orbit_id:?}, {obs_id:?})"
            ));
        }
        by_orbit
            .get_mut(orbit_id)
            .expect("orbit group inserted above")
            .insert(source_index, row);
    }
    if order.len() != orbits.len() {
        let referenced: HashSet<&str> = order.iter().map(String::as_str).collect();
        let missing = orbits
            .orbit_id
            .iter()
            .filter_map(|OrbitId(id)| (!referenced.contains(id.as_str())).then_some(id.as_str()))
            .collect::<Vec<_>>();
        return Err(format!(
            "orbits without explicit FittedOrbitMembers rows: {}",
            missing.join(", ")
        ));
    }

    Ok(order
        .into_iter()
        .map(|orbit_id| {
            let selected = &by_orbit[&orbit_id];
            let source_indices = (0..source_ids.len())
                .filter(|row| selected.contains_key(row))
                .collect::<Vec<_>>();
            let member_indices = source_indices
                .iter()
                .map(|row| selected[row])
                .collect::<Vec<_>>();
            OrbitMemberGroup {
                orbit_index: orbit_rows[orbit_id.as_str()],
                orbit_id,
                source_indices,
                member_indices,
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Frame;
    use crate::{
        CoordinateBatch, CoordinateValues, ObjectId, OriginArray, OriginId, TimeArray, TimeScale,
    };

    fn orbits(ids: &[&str]) -> OrbitBatch {
        OrbitBatch::new(
            ids.iter().map(|id| OrbitId((*id).to_string())).collect(),
            ids.iter()
                .map(|id| Some(ObjectId(format!("object-{id}"))))
                .collect(),
            CoordinateBatch::new(
                CoordinateValues::Cartesian(vec![[0.0; 6]; ids.len()]),
                Frame::Ecliptic,
                OriginArray::repeat(OriginId::Named("SUN".to_string()), ids.len()),
                Some(
                    TimeArray::new(
                        TimeScale::Tdb,
                        vec![crate::Epoch::new(60_000, 0); ids.len()],
                    )
                    .unwrap(),
                ),
                None,
            )
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn associations_preserve_first_group_and_source_row_order_with_overlap() {
        let ids = vec!["z".to_string(), "a".to_string(), "m".to_string()];
        let associations = AssociationBatch {
            detection_id: vec!["m".into(), "a".into(), "z".into(), "m".into()],
            object_id: vec![
                Some("g2".into()),
                Some("g1".into()),
                Some("g1".into()),
                Some("g1".into()),
            ],
        };
        let groups = plan_associations(&ids, &associations).unwrap();
        assert_eq!(groups[0].group_id, "g2");
        assert_eq!(groups[0].source_indices, vec![2]);
        assert_eq!(groups[1].group_id, "g1");
        assert_eq!(groups[1].source_indices, vec![0, 1, 2]);
    }

    #[test]
    fn associations_reject_null_unknown_duplicate_and_nonunique_sources() {
        let ids = vec!["a".to_string()];
        let mut associations = AssociationBatch {
            detection_id: vec!["a".into()],
            object_id: vec![None],
        };
        assert!(plan_associations(&ids, &associations)
            .unwrap_err()
            .contains("null object_id"));
        associations.object_id[0] = Some("g".into());
        associations.detection_id[0] = "missing".into();
        assert!(plan_associations(&ids, &associations)
            .unwrap_err()
            .contains("unknown"));
        associations.detection_id = vec!["a".into(), "a".into()];
        associations.object_id = vec![Some("g".into()), Some("g".into())];
        assert!(plan_associations(&ids, &associations)
            .unwrap_err()
            .contains("duplicate"));
        assert!(plan_associations(
            &["a".into(), "a".into()],
            &AssociationBatch {
                detection_id: vec![],
                object_id: vec![]
            }
        )
        .unwrap_err()
        .contains("not unique"));
    }

    #[test]
    fn empty_batches_have_no_groups() {
        let associations = AssociationBatch {
            detection_id: Vec::new(),
            object_id: Vec::new(),
        };
        assert!(plan_associations(&[], &associations).unwrap().is_empty());
        assert!(plan_orbit_members(&orbits(&[]), &[], &[], &[])
            .unwrap()
            .is_empty());
    }

    #[test]
    fn fitted_members_preserve_first_group_and_source_order() {
        let orbits = orbits(&["o1", "o2"]);
        let ids = vec!["z".to_string(), "a".to_string(), "m".to_string()];
        let groups = plan_orbit_members(
            &orbits,
            &ids,
            &["o2".into(), "o1".into(), "o1".into(), "o2".into()],
            &["m".into(), "m".into(), "z".into(), "a".into()],
        )
        .unwrap();
        assert_eq!(groups[0].orbit_id, "o2");
        assert_eq!(groups[0].source_indices, vec![1, 2]);
        assert_eq!(groups[0].member_indices, vec![3, 0]);
        assert_eq!(groups[1].source_indices, vec![0, 2]);
    }

    #[test]
    fn fitted_members_reject_missing_duplicate_and_unknown_references() {
        let ids = vec!["a".to_string(), "b".to_string()];
        let two = orbits(&["o1", "o2"]);
        let missing = plan_orbit_members(&two, &ids, &["o1".into()], &["a".into()]).unwrap_err();
        assert!(missing.contains("without explicit"));

        let one = orbits(&["o1"]);
        let duplicate = plan_orbit_members(
            &one,
            &ids,
            &["o1".into(), "o1".into()],
            &["a".into(), "a".into()],
        )
        .unwrap_err();
        assert!(duplicate.contains("duplicate"));

        let unknown_orbit =
            plan_orbit_members(&one, &ids, &["unknown".into()], &["a".into()]).unwrap_err();
        assert!(unknown_orbit.contains("unknown orbit_id"));

        let unknown_observation =
            plan_orbit_members(&one, &ids, &["o1".into()], &["unknown".into()]).unwrap_err();
        assert!(unknown_observation.contains("unknown obs_id"));
    }
}
