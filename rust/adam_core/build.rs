mod build_identity;

use build_identity::{resolve_commit, valid_commit};
use std::path::{Path, PathBuf};
use std::process::Command;

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

fn main() {
    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let supplied = std::env::var("ADAM_CORE_BUILD_COMMIT").ok();
    let (commit, checkout) = resolve_commit(&manifest, supplied)
        .unwrap_or_else(|message| panic!("cannot establish adam-core build identity: {message}"));
    if let Some(repo) = checkout.as_deref() {
        rerun_for_git_identity(repo);
    }
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
    println!("cargo:rerun-if-changed=build_identity.rs");
    println!("cargo:rerun-if-changed=.cargo_vcs_info.json");
}
