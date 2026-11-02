use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
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

fn release_attestation_path(repo: &PathBuf) -> Option<PathBuf> {
    let output = Command::new("git")
        .args([
            "-c",
            &format!("safe.directory={}", repo.display()),
            "rev-parse",
            "--git-path",
            "adam-core-release-attestation",
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

fn update_inventory_digest(digest: &mut Sha256, relative: &str, payload: &[u8]) {
    let encoded = relative.as_bytes();
    digest.update((encoded.len() as u64).to_be_bytes());
    digest.update(encoded);
    digest.update((payload.len() as u64).to_be_bytes());
    digest.update(payload);
}

fn source_python_files(root: &Path, repo: &Path) -> BTreeSet<String> {
    fn visit(directory: &Path, repo: &Path, output: &mut BTreeSet<String>) {
        let Ok(entries) = std::fs::read_dir(directory) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if path.file_name().and_then(|name| name.to_str()) != Some("__pycache__") {
                    visit(&path, repo, output);
                }
            } else if path.extension().and_then(|value| value.to_str()) == Some("py") {
                let Ok(relative) = path.strip_prefix(repo) else {
                    continue;
                };
                let value = relative.to_string_lossy().replace('\\', "/");
                if !value.split('/').any(|component| component == "tests")
                    && !path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| name.starts_with("test_"))
                {
                    output.insert(value);
                }
            }
        }
    }
    let mut output = BTreeSet::new();
    visit(root, repo, &mut output);
    output
}

fn validate_release_attestation(repo: &PathBuf, commit: &str) {
    let digest_env = std::env::var("ADAM_CORE_RELEASE_INVENTORY_DIGEST").expect(
        "release builds require ADAM_CORE_RELEASE_INVENTORY_DIGEST from the audited wrapper",
    );
    let nonce_env = std::env::var("ADAM_CORE_RELEASE_INVENTORY_NONCE").expect(
        "release builds require ADAM_CORE_RELEASE_INVENTORY_NONCE from the audited wrapper",
    );
    assert!(
        digest_env.len() == 64 && digest_env.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "release inventory digest must be 64 hexadecimal characters"
    );
    assert!(
        nonce_env.len() == 64 && nonce_env.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "release inventory nonce must be 64 hexadecimal characters"
    );
    let path = release_attestation_path(repo)
        .expect("release builds require a Git checkout and per-invocation attestation");
    let raw = std::fs::read_to_string(&path)
        .expect("release builds require a readable per-invocation attestation");
    let mut lines = raw.lines();
    assert_eq!(
        lines.next(),
        Some("adam-core-release-inventory-v2"),
        "unsupported release inventory format"
    );
    let attested_commit = lines
        .next()
        .and_then(|line| line.strip_prefix("commit="))
        .expect("release inventory is missing commit");
    let attested_nonce = lines
        .next()
        .and_then(|line| line.strip_prefix("nonce="))
        .expect("release inventory is missing nonce");
    let attested_digest = lines
        .next()
        .and_then(|line| line.strip_prefix("digest="))
        .expect("release inventory is missing digest");
    assert_eq!(attested_commit, commit, "release inventory commit is stale");
    assert_eq!(
        attested_nonce, nonce_env,
        "release inventory nonce is stale"
    );
    assert_eq!(
        attested_digest, digest_env,
        "release inventory digest environment is stale"
    );
    let files: Vec<String> = lines
        .map(|line| {
            line.strip_prefix("file=")
                .expect("release inventory contains a malformed file entry")
                .to_owned()
        })
        .collect();
    let mut sorted = files.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(
        files, sorted,
        "release inventory files must be unique and sorted"
    );

    let mut digest = Sha256::new();
    digest.update(b"adam-core-release-inventory-v2\0");
    for relative in &files {
        let payload = std::fs::read(repo.join(relative))
            .unwrap_or_else(|_| panic!("release inventory source is missing: {relative}"));
        update_inventory_digest(&mut digest, relative, &payload);
    }
    let actual = format!("{:x}", digest.finalize());
    assert_eq!(
        actual, digest_env,
        "release inventory source digest is stale"
    );

    let expected_python: BTreeSet<String> = files
        .iter()
        .filter(|path| {
            path.starts_with("src/adam_core/")
                && path.ends_with(".py")
                && !path.split('/').any(|component| component == "tests")
                && !Path::new(path)
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("test_"))
        })
        .cloned()
        .collect();
    let actual_python = source_python_files(&repo.join("src/adam_core"), repo);
    assert_eq!(
        actual_python, expected_python,
        "release package Python file set is stale or contains untracked source"
    );
    // A release attestation is single-use. Removing the watched file makes a
    // direct second release invocation rerun this script and fail closed;
    // the audited wrapper always creates a fresh nonce and file first. This
    // release-only missing sentinel never affects bare developer Cargo builds.
    println!("cargo:rerun-if-changed={}", path.display());
    std::fs::remove_file(&path)
        .expect("release attestation could not be consumed after validation");
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
    if std::env::var_os("ADAM_CORE_RELEASE_BUILD").is_some() {
        validate_release_attestation(&repo, &commit);
    }
    println!("cargo:rerun-if-env-changed=ADAM_CORE_RELEASE_INVENTORY_DIGEST");
    println!("cargo:rerun-if-env-changed=ADAM_CORE_RELEASE_INVENTORY_NONCE");

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
