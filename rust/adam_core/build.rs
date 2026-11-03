use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Command;

fn valid_commit(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn git_commit(repo: &Path) -> Option<String> {
    let output = Command::new("git")
        .args([
            "-c",
            &format!("safe.directory={}", repo.display()),
            "rev-parse",
            "HEAD",
        ])
        .current_dir(repo)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8(output.stdout).ok()?.trim().to_owned();
    valid_commit(&value).then_some(value)
}

fn git_dirty(repo: &Path) -> Option<bool> {
    let output = Command::new("git")
        .args([
            "-c",
            &format!("safe.directory={}", repo.display()),
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

fn emit_git_path_watch(repo: &Path, git_path: &str) {
    let Ok(output) = Command::new("git")
        .args([
            "-c",
            &format!("safe.directory={}", repo.display()),
            "rev-parse",
            "--git-path",
            git_path,
        ])
        .current_dir(repo)
        .output()
    else {
        return;
    };
    if output.status.success() {
        let value = String::from_utf8_lossy(&output.stdout).trim().to_owned();
        if !value.is_empty() {
            let path = PathBuf::from(value);
            let path = if path.is_absolute() {
                path
            } else {
                repo.join(path)
            };
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}

fn rerun_for_git_identity(repo: &Path) {
    for path in ["HEAD", "index", "packed-refs"] {
        emit_git_path_watch(repo, path);
    }
    if let Ok(output) = Command::new("git")
        .args([
            "-c",
            &format!("safe.directory={}", repo.display()),
            "symbolic-ref",
            "-q",
            "HEAD",
        ])
        .current_dir(repo)
        .output()
    {
        if output.status.success() {
            let reference = String::from_utf8_lossy(&output.stdout).trim().to_owned();
            if !reference.is_empty() {
                emit_git_path_watch(repo, &reference);
            }
        }
    }
}

fn packaged_commit(manifest: &Path) -> Option<String> {
    let raw = std::fs::read_to_string(manifest.join(".cargo_vcs_info.json")).ok()?;
    let value: Value = serde_json::from_str(&raw).ok()?;
    let commit = value.get("git")?.get("sha1")?.as_str()?.to_owned();
    valid_commit(&commit).then_some(commit)
}

fn main() {
    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let repo = manifest.join("../..").canonicalize().ok();
    if let Some(repo) = &repo {
        rerun_for_git_identity(repo);
    }
    let supplied = std::env::var("ADAM_CORE_BUILD_COMMIT").ok();
    if let Some(value) = &supplied {
        assert!(
            valid_commit(value),
            "ADAM_CORE_BUILD_COMMIT must be a 40-character lowercase hexadecimal commit"
        );
    }
    let discovered = repo.as_deref().and_then(git_commit);
    if let (Some(expected), Some(actual)) = (&supplied, &discovered) {
        assert_eq!(
            expected, actual,
            "ADAM_CORE_BUILD_COMMIT does not match source checkout"
        );
    }
    let commit = match (
        supplied,
        discovered,
        repo.as_deref().and_then(git_dirty),
        packaged_commit(&manifest),
    ) {
        (Some(commit), _, Some(false) | None, _) => commit,
        (Some(_), _, Some(true), _) => {
            panic!("release-supplied ADAM_CORE_BUILD_COMMIT cannot identify a dirty checkout")
        }
        (None, Some(commit), Some(true), _) => format!("dirty:{commit}"),
        (None, Some(commit), _, _) => commit,
        (None, None, _, Some(commit)) => commit,
        _ => "unknown".to_string(),
    };
    if std::env::var_os("ADAM_CORE_RELEASE_BUILD").is_some() {
        assert!(
            valid_commit(&commit),
            "release artifacts require an exact clean adam-core source commit"
        );
    }
    println!("cargo:rustc-env=ADAM_CORE_BUILD_COMMIT={commit}");
    println!("cargo:rerun-if-env-changed=ADAM_CORE_BUILD_COMMIT");
    println!("cargo:rerun-if-env-changed=ADAM_CORE_RELEASE_BUILD");
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=.cargo_vcs_info.json");
}
