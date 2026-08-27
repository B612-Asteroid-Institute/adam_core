//! Kernel/data-file resolution for pure-Rust adam-core consumers.
//!
//! See `README.md` for the discovery chain. The Python-hosted runtime keeps
//! passing pip-installed package paths and does not use this crate.

use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fmt;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

/// One resolvable data file, pinned to an exact published PyPI wheel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelSpec {
    /// Stable resolver id, also the env-override suffix (upper-cased).
    pub id: &'static str,
    /// Python module that ships the file (installed-Python probe).
    pub python_module: &'static str,
    /// Module attribute holding the file path in the Python package.
    pub python_attribute: &'static str,
    /// Exact immutable wheel URL on files.pythonhosted.org.
    pub wheel_url: &'static str,
    /// PyPI-published SHA-256 of the wheel file.
    pub wheel_sha256: &'static str,
    /// Path of the data member inside the wheel zip.
    pub member: &'static str,
    /// Cached output filename.
    pub filename: &'static str,
}

/// All adam-core data files, in `setup_SPICE` order (the six SPICE kernels),
/// then the obscodes table, then the ASSIST asteroid ephemeris.
pub const KERNEL_SPECS: &[KernelSpec] = &[
    KernelSpec {
        id: "leapseconds",
        python_module: "naif_leapseconds",
        python_attribute: "leapseconds",
        wheel_url: "https://files.pythonhosted.org/packages/52/fc/c0af39116e2894fc12dea442f60be7484dfa79c3dd728fb9cea43639c6ca/naif_leapseconds-2025.4.22-py3-none-any.whl",
        wheel_sha256: "abffc63a0d4f16bc99235152a550832092452abfd33b2a37a6467bfffb229d76",
        member: "naif_leapseconds/latest_leapseconds.tls",
        filename: "latest_leapseconds.tls",
    },
    KernelSpec {
        id: "de440",
        python_module: "naif_de440",
        python_attribute: "de440",
        wheel_url: "https://files.pythonhosted.org/packages/c6/56/e4e8b653c28b24dd3a64326548c166a9a1bc394f4c9c531b7c666ed71865/naif_de440-2020.12.21.1-py3-none-any.whl",
        wheel_sha256: "527193373382b157b4aa6820c9a90dd211df5fd2cb62725a18931183f47a91c4",
        member: "naif_de440/de440.bsp",
        filename: "de440.bsp",
    },
    KernelSpec {
        id: "eop_predict",
        python_module: "naif_eop_predict",
        python_attribute: "eop_predict",
        wheel_url: "https://files.pythonhosted.org/packages/26/f1/820f59da320c9290f387c4f98389df93eab85d7813692166ca9ef9ddc7b8/naif_eop_predict-2024.8.28.1-py3-none-any.whl",
        wheel_sha256: "d687e9ad01f0ba58c87d3f3e1b23236bed47c087e71b1358c0501afb023f9331",
        member: "naif_eop_predict/earth_200101_990827_predict.bpc",
        filename: "earth_200101_990827_predict.bpc",
    },
    KernelSpec {
        id: "eop_historical",
        python_module: "naif_eop_historical",
        python_attribute: "eop_historical",
        wheel_url: "https://files.pythonhosted.org/packages/70/b8/1dccea6f57b75ed58dd872d4ba6967c94533e51dc81cc84a67c684061ea4/naif_eop_historical-2024.8.28.1-py3-none-any.whl",
        wheel_sha256: "58f61cd0edcd303a41304dce2000e443c31c79ef097f2e78d1141e8528ecb05d",
        member: "naif_eop_historical/earth_620120_240827.bpc",
        filename: "earth_620120_240827.bpc",
    },
    KernelSpec {
        id: "eop_high_prec",
        python_module: "naif_eop_high_prec",
        python_attribute: "eop_high_prec",
        wheel_url: "https://files.pythonhosted.org/packages/d5/e7/43425a9153ad58a6c1bd717dd878459304d18edd9a969e6ac38b1d9ff509/naif_eop_high_prec-2026.4.30-py3-none-any.whl",
        wheel_sha256: "ee672cb229fdd668dc3c23bc77115c8209e6f8b4c754d578c217ae5bdb16c803",
        member: "naif_eop_high_prec/earth_latest_high_prec.bpc",
        filename: "earth_latest_high_prec.bpc",
    },
    KernelSpec {
        id: "earth_itrf93",
        python_module: "naif_earth_itrf93",
        python_attribute: "earth_itrf93",
        wheel_url: "https://files.pythonhosted.org/packages/51/af/cab609dbbc4636ee7db99f22479559fad0d9834f53329c96de35f74b0291/naif_earth_itrf93-2007.4.3.1-py3-none-any.whl",
        wheel_sha256: "3727e9e0d68004ce58374f9a3592dc006d958f8fb54cc2b1dc867e2846bfd5ff",
        member: "naif_earth_itrf93/earth_assoc_itrf93.tf",
        filename: "earth_assoc_itrf93.tf",
    },
    KernelSpec {
        id: "mpc_obscodes",
        python_module: "mpc_obscodes",
        python_attribute: "mpc_obscodes",
        wheel_url: "https://files.pythonhosted.org/packages/c4/78/e52e791fc41182f86633c1cfcc88efbe24f51b6c1fb217009c4e2a98cbdd/mpc_obscodes-2026.4.23-py3-none-any.whl",
        wheel_sha256: "aa71bec4032f7fc6b62f0842c42576b90e26cdc39459e4223d0cde5751e10d7c",
        member: "mpc_obscodes/obscodes_extended.json",
        filename: "obscodes_extended.json",
    },
    KernelSpec {
        id: "sb441_n16",
        python_module: "jpl_small_bodies_de441_n16",
        python_attribute: "de441_n16",
        wheel_url: "https://files.pythonhosted.org/packages/68/f1/9e493c4cc068f66b8691c51e9ea1b38cd1e451bdbd5b710b1a5c0442394d/jpl_small_bodies_de441_n16-2021.3.31.1-py3-none-any.whl",
        wheel_sha256: "43b7083bb1786a5dbf7a669b306b17d257dbd49694899f52be059d9ba3e34cfb",
        member: "jpl_small_bodies_de441_n16/sb441-n16.bsp",
        filename: "sb441-n16.bsp",
    },
];

/// The six kernels `adam_core.utils.spice.DEFAULT_KERNELS` furnshes, in order.
pub const DEFAULT_SPICE_KERNEL_IDS: &[&str] = &[
    "leapseconds",
    "de440",
    "eop_predict",
    "eop_historical",
    "eop_high_prec",
    "earth_itrf93",
];

#[derive(Debug)]
pub enum KernelDataError {
    UnknownKernel(String),
    Io {
        context: String,
        source: std::io::Error,
    },
    Fetch {
        url: String,
        message: String,
    },
    ChecksumMismatch {
        url: String,
        expected: String,
        actual: String,
    },
    WheelMember {
        wheel: String,
        member: String,
        message: String,
    },
    OfflineMiss {
        id: String,
    },
    NoCacheDir,
}

impl fmt::Display for KernelDataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownKernel(id) => write!(f, "unknown kernel id: {id}"),
            Self::Io { context, source } => write!(f, "{context}: {source}"),
            Self::Fetch { url, message } => write!(f, "failed to fetch {url}: {message}"),
            Self::ChecksumMismatch {
                url,
                expected,
                actual,
            } => write!(
                f,
                "checksum mismatch for {url}: expected {expected}, got {actual}"
            ),
            Self::WheelMember {
                wheel,
                member,
                message,
            } => write!(f, "failed to extract {member} from {wheel}: {message}"),
            Self::OfflineMiss { id } => write!(
                f,
                "kernel {id} is not available locally and ADAM_CORE_KERNEL_OFFLINE forbids fetching"
            ),
            Self::NoCacheDir => write!(
                f,
                "no cache directory: set ADAM_CORE_KERNEL_CACHE, XDG_CACHE_HOME, or HOME"
            ),
        }
    }
}

impl std::error::Error for KernelDataError {}

fn io_error(context: impl Into<String>) -> impl FnOnce(std::io::Error) -> KernelDataError {
    let context = context.into();
    move |source| KernelDataError::Io { context, source }
}

/// How the installed-Python probe locates an interpreter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PythonProbe {
    /// `ADAM_CORE_KERNEL_PYTHON`, else `$VIRTUAL_ENV/bin/python`, else `python3`.
    Auto,
    Explicit(PathBuf),
    Disabled,
}

/// Deterministic kernel resolver. `Resolver::from_env()` reads the documented
/// environment variables once; tests construct explicit configurations.
pub struct Resolver {
    overrides: HashMap<String, PathBuf>,
    python: PythonProbe,
    cache_root: Option<PathBuf>,
    offline: bool,
    probe_cache: OnceLock<HashMap<String, PathBuf>>,
}

impl Resolver {
    pub fn from_env() -> Self {
        let mut overrides = HashMap::new();
        for spec in KERNEL_SPECS {
            let var = format!("ADAM_CORE_KERNEL_{}", spec.id.to_uppercase());
            if let Some(path) = std::env::var_os(&var) {
                overrides.insert(spec.id.to_string(), PathBuf::from(path));
            }
        }
        let python = match std::env::var_os("ADAM_CORE_KERNEL_PYTHON") {
            Some(python) if python.is_empty() => PythonProbe::Disabled,
            Some(python) => PythonProbe::Explicit(PathBuf::from(python)),
            None => PythonProbe::Auto,
        };
        let cache_root = std::env::var_os("ADAM_CORE_KERNEL_CACHE")
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var_os("XDG_CACHE_HOME")
                    .map(|cache| PathBuf::from(cache).join("adam_core").join("kernels"))
            })
            .or_else(|| {
                std::env::var_os("HOME").map(|home| {
                    PathBuf::from(home)
                        .join(".cache")
                        .join("adam_core")
                        .join("kernels")
                })
            });
        let offline = std::env::var_os("ADAM_CORE_KERNEL_OFFLINE")
            .is_some_and(|value| !value.is_empty() && value != "0");
        Self::new(overrides, python, cache_root, offline)
    }

    pub fn new(
        overrides: HashMap<String, PathBuf>,
        python: PythonProbe,
        cache_root: Option<PathBuf>,
        offline: bool,
    ) -> Self {
        Self {
            overrides,
            python,
            cache_root,
            offline,
            probe_cache: OnceLock::new(),
        }
    }

    fn spec(id: &str) -> Result<&'static KernelSpec, KernelDataError> {
        KERNEL_SPECS
            .iter()
            .find(|spec| spec.id == id)
            .ok_or_else(|| KernelDataError::UnknownKernel(id.to_string()))
    }

    /// Resolve a kernel id to a readable local file path via the chain
    /// override -> installed-Python probe -> cache -> checksummed wheel fetch.
    pub fn resolve(&self, id: &str) -> Result<PathBuf, KernelDataError> {
        let spec = Self::spec(id)?;
        if let Some(path) = self.overrides.get(spec.id) {
            return Ok(path.clone());
        }
        if let Some(path) = self.python_paths().get(spec.id) {
            if path.is_file() {
                return Ok(path.clone());
            }
        }
        let cache_root = self.cache_root.clone().ok_or(KernelDataError::NoCacheDir)?;
        let destination = cache_root
            .join(spec.id)
            .join(&spec.wheel_sha256[..16])
            .join(spec.filename);
        if destination.is_file() {
            return Ok(destination);
        }
        if self.offline {
            return Err(KernelDataError::OfflineMiss {
                id: spec.id.to_string(),
            });
        }
        fetch_into_cache(spec, &destination)?;
        Ok(destination)
    }

    /// The six `DEFAULT_KERNELS` paths in `setup_SPICE` order.
    pub fn default_spice_kernel_paths(&self) -> Result<Vec<PathBuf>, KernelDataError> {
        DEFAULT_SPICE_KERNEL_IDS
            .iter()
            .map(|id| self.resolve(id))
            .collect()
    }

    /// The MPC observatory table as JSON content (what
    /// `AdamCoreSpiceBackend::load_mpc_obscodes` consumes).
    pub fn obscodes_json(&self) -> Result<String, KernelDataError> {
        let path = self.resolve("mpc_obscodes")?;
        std::fs::read_to_string(&path)
            .map_err(io_error(format!("failed to read {}", path.display())))
    }

    /// The (planets, asteroids) ephemeris paths for
    /// `assist_rs::Ephemeris::from_paths`.
    pub fn assist_ephemeris_paths(&self) -> Result<(PathBuf, PathBuf), KernelDataError> {
        Ok((self.resolve("de440")?, self.resolve("sb441_n16")?))
    }

    /// Furnsh the six default kernels and load the obscodes table: the
    /// pure-Rust equivalent of Python `setup_SPICE()` + `setup_mpc_obscodes()`.
    #[cfg(feature = "spice")]
    pub fn setup_spice_backend(
        &self,
        backend: &mut adam_core_rs_spice::AdamCoreSpiceBackend,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for path in self.default_spice_kernel_paths()? {
            backend.furnsh(&path)?;
        }
        backend.load_mpc_obscodes(&self.obscodes_json()?)?;
        Ok(())
    }

    fn python_paths(&self) -> &HashMap<String, PathBuf> {
        self.probe_cache.get_or_init(|| match &self.python {
            PythonProbe::Disabled => HashMap::new(),
            PythonProbe::Explicit(python) => probe_python(python),
            PythonProbe::Auto => {
                if let Some(venv) = std::env::var_os("VIRTUAL_ENV") {
                    let python = PathBuf::from(venv).join("bin").join("python");
                    if python.is_file() {
                        return probe_python(&python);
                    }
                }
                probe_python(Path::new("python3"))
            }
        })
    }
}

/// Import each data package in one interpreter run and print `id=path` lines,
/// tolerating packages that are not installed.
fn probe_script() -> String {
    let mut script = String::new();
    for spec in KERNEL_SPECS {
        script.push_str(&format!(
            "\ntry:\n    import {module}\n    print('{id}=' + str({module}.{attribute}))\nexcept Exception:\n    pass\n",
            module = spec.python_module,
            id = spec.id,
            attribute = spec.python_attribute,
        ));
    }
    script
}

fn probe_python(python: &Path) -> HashMap<String, PathBuf> {
    let output = match Command::new(python).arg("-c").arg(probe_script()).output() {
        Ok(output) if output.status.success() => output,
        _ => return HashMap::new(),
    };
    parse_probe_output(&String::from_utf8_lossy(&output.stdout))
}

fn parse_probe_output(stdout: &str) -> HashMap<String, PathBuf> {
    stdout
        .lines()
        .filter_map(|line| {
            let (id, path) = line.split_once('=')?;
            let path = PathBuf::from(path.trim());
            KERNEL_SPECS
                .iter()
                .any(|spec| spec.id == id)
                .then(|| (id.to_string(), path))
        })
        .collect()
}

/// Download the pinned wheel, verify its SHA-256 while streaming to a
/// temporary file, extract the data member, and atomically publish it.
fn fetch_into_cache(spec: &KernelSpec, destination: &Path) -> Result<(), KernelDataError> {
    let parent = destination.parent().ok_or(KernelDataError::NoCacheDir)?;
    std::fs::create_dir_all(parent)
        .map_err(io_error(format!("failed to create {}", parent.display())))?;

    let response = ureq::get(spec.wheel_url)
        .timeout(std::time::Duration::from_secs(3600))
        .call()
        .map_err(|err| KernelDataError::Fetch {
            url: spec.wheel_url.to_string(),
            message: err.to_string(),
        })?;
    let mut reader = response.into_reader();

    let wheel_path = destination.with_extension("wheel.partial");
    let mut wheel_file = std::fs::File::create(&wheel_path).map_err(io_error(format!(
        "failed to create {}",
        wheel_path.display()
    )))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(io_error(format!("failed reading {}", spec.wheel_url)))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        wheel_file
            .write_all(&buffer[..read])
            .map_err(io_error(format!("failed writing {}", wheel_path.display())))?;
    }
    wheel_file.flush().map_err(io_error(format!(
        "failed flushing {}",
        wheel_path.display()
    )))?;
    drop(wheel_file);

    let actual = format!("{:x}", hasher.finalize());
    if actual != spec.wheel_sha256 {
        let _ = std::fs::remove_file(&wheel_path);
        return Err(KernelDataError::ChecksumMismatch {
            url: spec.wheel_url.to_string(),
            expected: spec.wheel_sha256.to_string(),
            actual,
        });
    }

    let result = extract_member_atomic(&wheel_path, spec.member, destination);
    let _ = std::fs::remove_file(&wheel_path);
    result
}

/// Extract `member` from a wheel (zip) file into `destination` via a
/// temporary sibling and rename, so concurrent resolvers never observe a
/// partial kernel.
fn extract_member_atomic(
    wheel_path: &Path,
    member: &str,
    destination: &Path,
) -> Result<(), KernelDataError> {
    let wheel_error = |message: String| KernelDataError::WheelMember {
        wheel: wheel_path.display().to_string(),
        member: member.to_string(),
        message,
    };
    let wheel_file = std::fs::File::open(wheel_path)
        .map_err(io_error(format!("failed to open {}", wheel_path.display())))?;
    let mut archive =
        zip::ZipArchive::new(wheel_file).map_err(|err| wheel_error(err.to_string()))?;
    let mut entry = archive
        .by_name(member)
        .map_err(|err| wheel_error(err.to_string()))?;

    let staged = destination.with_extension("partial");
    let mut staged_file = std::fs::File::create(&staged)
        .map_err(io_error(format!("failed to create {}", staged.display())))?;
    std::io::copy(&mut entry, &mut staged_file).map_err(|err| wheel_error(err.to_string()))?;
    staged_file
        .flush()
        .map_err(io_error(format!("failed flushing {}", staged.display())))?;
    drop(staged_file);
    std::fs::rename(&staged, destination).map_err(io_error(format!(
        "failed to publish {}",
        destination.display()
    )))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_fake_wheel(path: &Path, member: &str, contents: &[u8]) {
        let file = std::fs::File::create(path).unwrap();
        let mut wheel = zip::ZipWriter::new(file);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        wheel.start_file(member, options).unwrap();
        wheel.write_all(contents).unwrap();
        wheel.finish().unwrap();
    }

    #[test]
    fn manifest_is_well_formed() {
        assert_eq!(KERNEL_SPECS.len(), 8);
        for spec in KERNEL_SPECS {
            assert_eq!(spec.wheel_sha256.len(), 64, "{}", spec.id);
            assert!(
                spec.wheel_sha256.chars().all(|c| c.is_ascii_hexdigit()),
                "{}",
                spec.id
            );
            assert!(
                spec.wheel_url
                    .starts_with("https://files.pythonhosted.org/packages/"),
                "{}",
                spec.id
            );
            assert!(spec.member.ends_with(spec.filename), "{}", spec.id);
        }
        for id in DEFAULT_SPICE_KERNEL_IDS {
            assert!(KERNEL_SPECS.iter().any(|spec| spec.id == *id));
        }
    }

    #[test]
    fn explicit_override_wins() {
        let dir = tempfile::tempdir().unwrap();
        let kernel = dir.path().join("custom.bsp");
        std::fs::write(&kernel, b"DAF/SPK fake").unwrap();
        let resolver = Resolver::new(
            HashMap::from([("de440".to_string(), kernel.clone())]),
            PythonProbe::Disabled,
            None,
            true,
        );
        assert_eq!(resolver.resolve("de440").unwrap(), kernel);
    }

    #[test]
    fn python_probe_paths_are_used_when_files_exist() {
        let dir = tempfile::tempdir().unwrap();
        let kernel = dir.path().join("latest_leapseconds.tls");
        std::fs::write(&kernel, b"KPL/LSK").unwrap();
        // Fake interpreter: a script that prints the probe line.
        let python = dir.path().join("python");
        std::fs::write(
            &python,
            format!("#!/bin/sh\necho 'leapseconds={}'\n", kernel.display()),
        )
        .unwrap();
        let mut permissions = std::fs::metadata(&python).unwrap().permissions();
        std::os::unix::fs::PermissionsExt::set_mode(&mut permissions, 0o755);
        std::fs::set_permissions(&python, permissions).unwrap();

        let resolver = Resolver::new(HashMap::new(), PythonProbe::Explicit(python), None, true);
        assert_eq!(resolver.resolve("leapseconds").unwrap(), kernel);
    }

    #[test]
    fn cache_hit_avoids_fetch_and_offline_miss_fails() {
        let dir = tempfile::tempdir().unwrap();
        let resolver = Resolver::new(
            HashMap::new(),
            PythonProbe::Disabled,
            Some(dir.path().to_path_buf()),
            true,
        );
        let spec = KERNEL_SPECS.iter().find(|s| s.id == "leapseconds").unwrap();
        let cached = dir
            .path()
            .join(spec.id)
            .join(&spec.wheel_sha256[..16])
            .join(spec.filename);
        std::fs::create_dir_all(cached.parent().unwrap()).unwrap();
        std::fs::write(&cached, b"KPL/LSK cached").unwrap();
        assert_eq!(resolver.resolve("leapseconds").unwrap(), cached);

        let miss = resolver.resolve("de440").unwrap_err();
        assert!(matches!(miss, KernelDataError::OfflineMiss { .. }));
    }

    #[test]
    fn wheel_member_extraction_is_atomic_and_correct() {
        let dir = tempfile::tempdir().unwrap();
        let wheel = dir.path().join("fake.whl");
        write_fake_wheel(&wheel, "pkg/data.tls", b"KPL/LSK payload");
        let destination = dir.path().join("out").join("data.tls");
        std::fs::create_dir_all(destination.parent().unwrap()).unwrap();
        extract_member_atomic(&wheel, "pkg/data.tls", &destination).unwrap();
        assert_eq!(std::fs::read(&destination).unwrap(), b"KPL/LSK payload");
        assert!(!destination.with_extension("partial").exists());

        let missing = extract_member_atomic(&wheel, "pkg/absent.tls", &dir.path().join("never"));
        assert!(matches!(missing, Err(KernelDataError::WheelMember { .. })));
    }

    #[test]
    fn probe_output_parsing_ignores_unknown_ids_and_noise() {
        let parsed = parse_probe_output(
            "leapseconds=/a/b.tls\nbogus=/nope\nnot a line\nde440=/c/de440.bsp\n",
        );
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed["leapseconds"], PathBuf::from("/a/b.tls"));
        assert_eq!(parsed["de440"], PathBuf::from("/c/de440.bsp"));
    }

    #[test]
    fn unknown_kernel_id_fails_loudly() {
        let resolver = Resolver::new(HashMap::new(), PythonProbe::Disabled, None, true);
        assert!(matches!(
            resolver.resolve("nonexistent"),
            Err(KernelDataError::UnknownKernel(_))
        ));
    }

    /// End-to-end integration gate: resolve the default kernels (from env,
    /// typically via the installed-Python probe) and drive the Rust SPICE
    /// backend exactly like Python `setup_SPICE()` + `Observers.from_codes`.
    #[cfg(feature = "spice")]
    #[test]
    #[ignore = "requires resolvable kernel data (Python env, cache, or network)"]
    fn live_setup_spice_backend_serves_states_and_obscodes() {
        let resolver = Resolver::from_env();
        let mut backend = adam_core_rs_spice::AdamCoreSpiceBackend::new();
        resolver.setup_spice_backend(&mut backend).unwrap();
        assert!(backend.mpc_obscodes_loaded() > 500);
        // Earth (399) relative to the Sun (10) in J2000 at ET=0.
        let state = backend.spkez(399, 0.0, "J2000", 10).unwrap();
        let distance_km = (state[0] * state[0] + state[1] * state[1] + state[2] * state[2]).sqrt();
        assert!((1.3e8..1.6e8).contains(&distance_km), "{distance_km}");
    }

    /// Live network gate: fetches the smallest pinned wheel (~9.5 KB) and
    /// verifies checksum + extraction end-to-end.
    #[test]
    #[ignore = "performs a live PyPI download; run explicitly"]
    fn live_fetch_of_leapseconds_wheel_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let resolver = Resolver::new(
            HashMap::new(),
            PythonProbe::Disabled,
            Some(dir.path().to_path_buf()),
            false,
        );
        let path = resolver.resolve("leapseconds").unwrap();
        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(contents.starts_with("KPL/LSK"));
        // Second resolve is a pure cache hit.
        assert_eq!(resolver.resolve("leapseconds").unwrap(), path);
    }
}
