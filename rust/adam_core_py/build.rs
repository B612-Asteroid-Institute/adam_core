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

fn relevant_untracked_path(path: &str) -> bool {
    (path.starts_with("src/adam_core/") && !path.contains("/tests/") && path.ends_with(".py"))
        || (path.starts_with("rust/")
            && (path.contains("/src/")
                || path.ends_with("/Cargo.toml")
                || path.ends_with("/build.rs")))
}

fn relevant_untracked_files(repo: &PathBuf) -> Option<Vec<String>> {
    let output = Command::new("git")
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
    if !output.status.success() {
        return None;
    }
    let mut relevant = Vec::new();
    for raw in output
        .stdout
        .split(|byte| *byte == 0)
        .filter(|raw| !raw.is_empty())
    {
        let path = std::str::from_utf8(raw).ok()?;
        if relevant_untracked_path(path) {
            relevant.push(path.to_owned());
        }
    }
    relevant.sort_unstable();
    Some(relevant)
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
    // make the build dirty. Tests, caches, wheelhouse/, documentation output,
    // generated extensions, and the nested Assist checkout do not.
    relevant_untracked_files(repo).map(|paths| !paths.is_empty())
}

fn source_inventory_path(repo: &PathBuf) -> Option<PathBuf> {
    let output = Command::new("git")
        .args([
            "-c",
            &format!("safe.directory={}", repo.display()),
            "rev-parse",
            "--git-path",
            "adam-core-source-inventory",
        ])
        .current_dir(repo)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let raw = String::from_utf8(output.stdout).ok()?;
    let path = PathBuf::from(raw.trim());
    if path.as_os_str().is_empty() {
        None
    } else if path.is_absolute() {
        Some(path)
    } else {
        Some(repo.join(path))
    }
}

fn emit_source_inventory_watch(repo: &PathBuf) {
    if let Some(path) = source_inventory_path(repo) {
        println!("cargo:rerun-if-changed={}", path.display());
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
    // The wheel-version pre-step refreshes this narrow inventory only when
    // relevant new runtime source changes. This avoids recursive source-root
    // watches and their generated-extension/__pycache__ rebuild loop.
    emit_source_inventory_watch(&repo);

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
