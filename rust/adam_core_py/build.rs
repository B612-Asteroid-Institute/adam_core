use std::path::{Path, PathBuf};
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
    // Tracked edits/staging anywhere in Core affect the source identity. The
    // exact adam-assist checkout used by the release workflow is a separate
    // repository and is deliberately outside Core's identity.
    let tracked = Command::new("git")
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
    if !tracked.status.success() {
        return None;
    }
    if !tracked.stdout.is_empty() {
        return Some(true);
    }

    // Only untracked files that can enter the Python package or a Rust crate
    // make the build dirty. Caches, wheelhouse/, docs output and the nested
    // Assist checkout cannot affect the compiled/native package.
    let untracked = Command::new("git")
        .args([
            "-c",
            &format!("safe.directory={}", repo.display()),
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            "src/adam_core",
            "rust",
        ])
        .current_dir(repo)
        .output()
        .ok()?;
    if !untracked.status.success() {
        return None;
    }
    let relevant = untracked.stdout.split(|byte| *byte == 0).any(|raw| {
        let Ok(path) = std::str::from_utf8(raw) else {
            return true;
        };
        path.starts_with("src/adam_core/")
            || (path.starts_with("rust/")
                && (path.contains("/src/")
                    || path.ends_with("/Cargo.toml")
                    || path.ends_with("/build.rs")))
    });
    Some(relevant)
}

fn emit_source_directory_watches(repo: &Path) {
    // Directory watches are needed for additions/removals: a newly-created
    // untracked source file cannot be listed by an earlier build-script run.
    // Restrict them to source roots so target/, wheelhouse/ and the nested
    // adam-assist checkout never cause rebuild loops.
    let python = repo.join("src/adam_core");
    if python.is_dir() {
        println!("cargo:rerun-if-changed={}", python.display());
    }
    let rust = repo.join("rust");
    if rust.is_dir() {
        println!("cargo:rerun-if-changed={}", rust.display());
    }
}

fn emit_git_path_watch(repo: &PathBuf, git_path: &str) {
    if let Ok(output) = Command::new("git")
        .args([
            "-c",
            &format!("safe.directory={}", repo.display()),
            "rev-parse",
            "--git-path",
            git_path,
        ])
        .current_dir(repo)
        .output()
    {
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
    // Every tracked distribution input is an explicit dependency. This
    // avoids stale clean identities after commits or edits outside a short
    // hand-maintained source list, without recursively watching `target/`.
    if let Ok(output) = Command::new("git")
        .args([
            "-c",
            &format!("safe.directory={}", repo.display()),
            "ls-files",
            "-z",
        ])
        .current_dir(&repo)
        .output()
    {
        if output.status.success() {
            for relative in output
                .stdout
                .split(|byte| *byte == 0)
                .filter(|p| !p.is_empty())
            {
                if let Ok(relative) = std::str::from_utf8(relative) {
                    println!("cargo:rerun-if-changed={}", repo.join(relative).display());
                }
            }
        }
    }
    emit_source_directory_watches(&repo);

    // The index detects staged paths and the current branch ref detects a
    // commit even though `.git/HEAD` still contains the same symbolic name.
    for git_path in ["HEAD", "index", "packed-refs"] {
        emit_git_path_watch(&repo, git_path);
    }
    if let Ok(output) = Command::new("git")
        .args([
            "-c",
            &format!("safe.directory={}", repo.display()),
            "symbolic-ref",
            "-q",
            "HEAD",
        ])
        .current_dir(&repo)
        .output()
    {
        if output.status.success() {
            let reference = String::from_utf8_lossy(&output.stdout).trim().to_owned();
            if !reference.is_empty() {
                emit_git_path_watch(&repo, &reference);
            }
        }
    }
    // The release-matrix `adam-assist/` checkout is deliberately excluded;
    // its exact SHA is verified independently by that workflow.
}
