use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Command;

pub fn valid_commit(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn git_output(repo: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .arg("-c")
        .arg(format!("safe.directory={}", repo.display()))
        .args(args)
        .current_dir(repo)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8(output.stdout).ok()?.trim().to_owned())
}

/// Return the adam-core checkout root only when the manifest has the canonical
/// in-repository location and Git itself reports that exact root.
pub fn verified_checkout(manifest: &Path) -> Option<PathBuf> {
    let manifest = manifest.canonicalize().ok()?;
    let expected_root = manifest.join("../..").canonicalize().ok()?;
    let expected_manifest = expected_root.join("rust/adam_core").canonicalize().ok()?;
    if manifest != expected_manifest {
        return None;
    }
    let top = PathBuf::from(git_output(
        &expected_root,
        &["rev-parse", "--show-toplevel"],
    )?);
    if top.canonicalize().ok()? != expected_root {
        return None;
    }
    let tracked = Command::new("git")
        .arg("-c")
        .arg(format!("safe.directory={}", expected_root.display()))
        .args([
            "ls-files",
            "--error-unmatch",
            "--",
            "rust/adam_core/Cargo.toml",
            "rust/adam_core/src/lib.rs",
        ])
        .current_dir(&expected_root)
        .output()
        .ok()?;
    tracked.status.success().then_some(expected_root)
}

pub fn git_commit(repo: &Path) -> Option<String> {
    let value = git_output(repo, &["rev-parse", "HEAD"])?;
    valid_commit(&value).then_some(value)
}

pub fn git_dirty(repo: &Path) -> Option<bool> {
    let output = Command::new("git")
        .arg("-c")
        .arg(format!("safe.directory={}", repo.display()))
        .args([
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--",
            ".",
            ":(exclude)adam-assist",
        ])
        .current_dir(repo)
        .output()
        .ok()?;
    output.status.success().then_some(!output.stdout.is_empty())
}

/// Read Cargo's immutable package identity. Presence is authoritative: invalid
/// JSON, a malformed commit, or `dirty: true` fails closed rather than falling
/// through to an enclosing consumer repository.
pub fn packaged_commit(manifest: &Path) -> Result<Option<String>, String> {
    let path = manifest.join(".cargo_vcs_info.json");
    let raw = match std::fs::read_to_string(&path) {
        Ok(raw) => raw,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("cannot read {}: {error}", path.display())),
    };
    let value: Value = serde_json::from_str(&raw)
        .map_err(|error| format!("invalid {}: {error}", path.display()))?;
    let git = value
        .get("git")
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{} has no git identity", path.display()))?;
    if git.get("dirty").and_then(Value::as_bool).unwrap_or(false) {
        return Err(format!("{} records dirty packaged sources", path.display()));
    }
    let commit = git
        .get("sha1")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{} has no git.sha1", path.display()))?;
    if !valid_commit(commit) {
        return Err(format!("{} has an invalid git.sha1", path.display()));
    }
    Ok(Some(commit.to_owned()))
}

pub fn resolve_commit(
    manifest: &Path,
    supplied: Option<String>,
) -> Result<(String, Option<PathBuf>), String> {
    if let Some(value) = &supplied {
        if !valid_commit(value) {
            return Err(
                "ADAM_CORE_BUILD_COMMIT must be a 40-character lowercase hexadecimal commit"
                    .to_string(),
            );
        }
    }

    let packaged = packaged_commit(manifest)?;
    if let Some(packaged) = packaged {
        if let Some(expected) = &supplied {
            if expected != &packaged {
                return Err(
                    "ADAM_CORE_BUILD_COMMIT does not match packaged source identity".to_string(),
                );
            }
        }
        return Ok((supplied.unwrap_or(packaged), None));
    }

    let checkout = verified_checkout(manifest);
    let discovered = checkout.as_deref().and_then(git_commit);
    if let (Some(expected), Some(actual)) = (&supplied, &discovered) {
        if expected != actual {
            return Err("ADAM_CORE_BUILD_COMMIT does not match source checkout".to_string());
        }
    }
    let dirty = checkout.as_deref().and_then(git_dirty);
    let commit = match (supplied, discovered, dirty) {
        (Some(commit), _, Some(false) | None) => commit,
        (Some(_), _, Some(true)) => {
            return Err(
                "release-supplied ADAM_CORE_BUILD_COMMIT cannot identify a dirty checkout"
                    .to_string(),
            )
        }
        (None, Some(commit), Some(true)) => format!("dirty:{commit}"),
        (None, Some(commit), _) => commit,
        (None, None, _) => "unknown".to_string(),
    };
    Ok((commit, checkout))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT: AtomicU64 = AtomicU64::new(0);
    const CORE_COMMIT: &str = "06419201e6f8749ea9d9098d64945653b03c63db";

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            let id = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "adam-core-build-identity-{name}-{}-{id}",
                std::process::id()
            ));
            let _ = fs::remove_dir_all(&path);
            fs::create_dir_all(&path).expect("temporary directory");
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn write_package(manifest: &Path, dirty: bool) {
        fs::create_dir_all(manifest.join("src")).expect("package directories");
        fs::write(
            manifest.join("Cargo.toml"),
            "[package]\nname='adam_core'\nversion='0.7.0'\n",
        )
        .expect("manifest");
        fs::write(manifest.join("src/lib.rs"), "pub fn marker() {}\n").expect("source");
        fs::write(
            manifest.join(".cargo_vcs_info.json"),
            format!(r#"{{"git":{{"sha1":"{CORE_COMMIT}","dirty":{dirty}}},"path_in_vcs":"rust/adam_core"}}"#),
        )
        .expect("vcs info");
    }

    fn git(repo: &Path, args: &[&str]) {
        let status = Command::new("git")
            .args(args)
            .current_dir(repo)
            .status()
            .expect("git command");
        assert!(status.success(), "git {args:?}");
    }

    fn init_consumer(repo: &Path) -> String {
        fs::create_dir_all(repo).expect("consumer");
        git(repo, &["init", "-q"]);
        git(repo, &["config", "user.name", "Adam Core Test"]);
        git(repo, &["config", "user.email", "test@example.invalid"]);
        fs::write(repo.join("README"), "consumer\n").expect("readme");
        git(repo, &["add", "."]);
        git(repo, &["commit", "-qm", "consumer"]);
        git_output(repo, &["rev-parse", "HEAD"]).expect("consumer commit")
    }

    #[test]
    fn registry_package_uses_cargo_vcs_identity() {
        let temp = TempDir::new("registry");
        let manifest = temp.0.join("adam_core-0.7.0");
        write_package(&manifest, false);
        let (commit, checkout) = resolve_commit(&manifest, None).expect("package identity");
        assert_eq!(commit, CORE_COMMIT);
        assert!(checkout.is_none());
    }

    #[test]
    fn vendored_package_ignores_clean_unrelated_repository() {
        let temp = TempDir::new("vendor-clean");
        let consumer_commit = init_consumer(&temp.0);
        let manifest = temp.0.join("vendor/adam_core-0.7.0");
        write_package(&manifest, false);
        git(&temp.0, &["add", "."]);
        git(&temp.0, &["commit", "-qm", "vendor"]);
        let (commit, checkout) = resolve_commit(&manifest, None).expect("vendored identity");
        assert_eq!(commit, CORE_COMMIT);
        assert_ne!(commit, consumer_commit);
        assert!(checkout.is_none());
    }

    #[test]
    fn vendored_package_ignores_dirty_unrelated_repository() {
        let temp = TempDir::new("vendor-dirty");
        init_consumer(&temp.0);
        let manifest = temp.0.join("vendor/adam_core-0.7.0");
        write_package(&manifest, false);
        fs::write(temp.0.join("README"), "dirty consumer\n").expect("dirty consumer");
        let (commit, checkout) = resolve_commit(&manifest, None).expect("vendored identity");
        assert_eq!(commit, CORE_COMMIT);
        assert!(checkout.is_none());
    }

    #[test]
    fn dirty_package_vcs_info_is_rejected() {
        let temp = TempDir::new("package-dirty");
        let manifest = temp.0.join("adam_core-0.7.0");
        write_package(&manifest, true);
        assert!(resolve_commit(&manifest, None)
            .expect_err("dirty package must fail")
            .contains("dirty packaged sources"));
    }

    #[test]
    fn normal_checkout_records_its_own_head() {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        if manifest.join(".cargo_vcs_info.json").exists() {
            return;
        }
        let root = verified_checkout(&manifest).expect("canonical checkout");
        let expected = git_commit(&root).expect("checkout commit");
        let (actual, checkout) = resolve_commit(&manifest, None).expect("checkout identity");
        assert_eq!(actual.trim_start_matches("dirty:"), expected);
        assert_eq!(checkout.as_deref(), Some(root.as_path()));
    }
}
