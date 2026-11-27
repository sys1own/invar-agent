use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
    println!("cargo:rerun-if-env-changed=PYTHON_SYS_EXECUTABLE");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "windows" {
        return;
    }

    // Locate the Python interpreter so Windows `cargo test` binaries can resolve python3XX.dll
    // without requiring callers to manually mutate PATH.
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(p) = env::var("PYO3_PYTHON") {
        candidates.push(PathBuf::from(p));
    }
    if let Ok(p) = env::var("PYTHON_SYS_EXECUTABLE") {
        candidates.push(PathBuf::from(p));
    }
    if let Ok(manifest_dir) = env::var("CARGO_MANIFEST_DIR") {
        let repo_root = PathBuf::from(manifest_dir)
            .parent()
            .and_then(|p| p.parent())
            .map(PathBuf::from);
        if let Some(root) = repo_root {
            let venv_py = root.join(".venv").join("Scripts").join("python.exe");
            if venv_py.is_file() {
                candidates.push(venv_py);
            }
        }
    }
    candidates.push(PathBuf::from("python"));
    candidates.push(PathBuf::from("python3"));

    for py in candidates {
        if let Ok(out) = Command::new(&py)
            .args([
                "-c",
                "import sys, os; print(os.path.abspath(sys.base_prefix))",
            ])
            .output()
        {
            if out.status.success() {
                let dir_str = String::from_utf8_lossy(&out.stdout).trim().to_string();
                let base_dir = PathBuf::from(&dir_str);
                if !dir_str.is_empty() && base_dir.is_dir() {
                    println!("cargo:rustc-link-search=native={}", dir_str);
                    let libs_dir = base_dir.join("libs");
                    if libs_dir.is_dir() {
                        println!("cargo:rustc-link-search=native={}", libs_dir.display());
                    }
                    // Copy python3*.dll into target/<profile>/deps so `cargo test` executables
                    // find the DLL next to the test binary on Windows.
                    if let Ok(out_dir) = env::var("OUT_DIR") {
                        let out_path = PathBuf::from(out_dir);
                        if let Some(profile_dir) = out_path
                            .parent()
                            .and_then(|p| p.parent())
                            .and_then(|p| p.parent())
                        {
                            let deps_dir = profile_dir.join("deps");
                            let _ = fs::create_dir_all(&deps_dir);
                            if let Ok(entries) = fs::read_dir(&base_dir) {
                                for entry in entries.flatten() {
                                    let name = entry.file_name();
                                    let name_str = name.to_string_lossy().to_ascii_lowercase();
                                    if name_str.starts_with("python3") && name_str.ends_with(".dll") {
                                        let dest = deps_dir.join(&name);
                                        if !dest.exists() {
                                            let _ = fs::copy(entry.path(), dest);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
            }
        }
    }
}
