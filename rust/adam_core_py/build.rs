use std::path::PathBuf;
use std::process::Command;

fn git_commit(repo: &PathBuf) -> Option<String> {
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
    (value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())).then_some(value)
}

fn git_dirty(repo: &PathBuf) -> Option<bool> {
    let output = Command::new("git")
        .args([
            "-c",
            &format!("safe.directory={}", repo.display()),
            "status",
            "--porcelain",
            "--untracked-files=normal",
        ])
        .current_dir(repo)
        .output()
        .ok()?;
    output.status.success().then_some(!output.stdout.is_empty())
}

fn main() {
    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let repo = manifest
        .join("../..")
        .canonicalize()
        .expect("repository path");
    let discovered = git_commit(&repo);
    let supplied = std::env::var("ADAM_CORE_BUILD_COMMIT").ok();
    if let Some(expected) = &supplied {
        assert!(
            expected.len() == 40
                && expected
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
            "ADAM_CORE_BUILD_COMMIT must be a 40-character lowercase hexadecimal commit"
        );
    }
    if let (Some(expected), Some(actual)) = (&supplied, &discovered) {
        assert_eq!(
            expected, actual,
            "ADAM_CORE_BUILD_COMMIT does not match source checkout"
        );
    }
    let commit = match (supplied, discovered, git_dirty(&repo)) {
        (Some(commit), _, Some(false)) => commit,
        (Some(_), _, Some(true)) => {
            panic!("release-supplied ADAM_CORE_BUILD_COMMIT cannot identify a dirty checkout")
        }
        (Some(commit), _, None) => commit,
        (None, Some(commit), Some(true)) => format!("dirty:{commit}"),
        (None, Some(commit), _) => commit,
        (None, None, _) => "unknown".to_string(),
    };
    if std::env::var_os("ADAM_CORE_RELEASE_BUILD").is_some() {
        assert!(
            commit.len() == 40
                && commit
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
            "release artifacts require an exact clean adam-core source commit"
        );
    }
    println!("cargo:rustc-env=ADAM_CORE_BUILD_COMMIT={commit}");
    println!("cargo:rerun-if-env-changed=ADAM_CORE_BUILD_COMMIT");
    println!("cargo:rerun-if-env-changed=ADAM_CORE_RELEASE_BUILD");
    // Re-run the identity check when any source that contributes to the
    // mixed Python/Rust distribution changes. Cargo otherwise caches build
    // script output across ordinary working-tree edits.
    for relative in [
        "Cargo.toml",
        "Cargo.lock",
        "pyproject.toml",
        "pdm.lock",
        "CHANGELOG.md",
        "src",
        "rust",
        "migration",
        "docs",
    ] {
        let path = repo.join(relative);
        if path.exists() {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
    // Worktrees store `.git` as a redirection file, so ask Git for the
    // actual HEAD path rather than assuming `.git/HEAD` is a directory.
    if let Ok(output) = Command::new("git")
        .args([
            "-c",
            &format!("safe.directory={}", repo.display()),
            "rev-parse",
            "--git-path",
            "HEAD",
        ])
        .current_dir(&repo)
        .output()
    {
        if output.status.success() {
            let head = String::from_utf8_lossy(&output.stdout).trim().to_owned();
            if !head.is_empty() {
                let path = PathBuf::from(&head);
                let path = if path.is_absolute() {
                    path
                } else {
                    repo.join(path)
                };
                println!("cargo:rerun-if-changed={}", path.display());
            }
        }
    }
}
