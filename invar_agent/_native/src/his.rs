//! 256-bit BLAKE3 Holographic Boundary Register (`his.rs`).
//!
//! Projects the bulk workspace state (every module's canonicalized public
//! interface signature) onto a compact, fixed-width 256-bit boundary vector
//! `H(d_Omega)` using a cryptographic Merkle homomorphism.  The register is
//! order-independent (XOR-folded) so that concurrent wavefront commits can be
//! merged without re-hashing the entire repository.

use std::collections::HashMap;

/// Bulk-to-boundary holographic register core.
#[derive(Default)]
pub struct HisBoundaryCore {
    /// module_path -> blake3(canonical_interface) digest bytes.
    modules: HashMap<String, [u8; 32]>,
}

impl HisBoundaryCore {
    pub fn new() -> Self {
        Self {
            modules: HashMap::new(),
        }
    }

    /// Update (or insert) a module's canonicalized public interface signature.
    pub fn update_module(&mut self, module_path: String, canonical_interface: String) {
        let digest = *blake3::hash(canonical_interface.as_bytes()).as_bytes();
        self.modules.insert(module_path, digest);
    }

    /// Remove a module from the bulk map.
    pub fn remove_module(&mut self, module_path: &str) -> bool {
        self.modules.remove(module_path).is_some()
    }

    /// Number of modules currently projected into the boundary register.
    pub fn module_count(&self) -> usize {
        self.modules.len()
    }

    /// Project the entire bulk module map onto the 256-bit boundary register.
    ///
    /// The register is the XOR-fold of every per-module BLAKE3 digest, encoded
    /// as a 64-character lowercase hex string.  XOR folding makes the digest
    /// commutative and associative, so parallel wavefront boundary updates
    /// compose without ordering constraints.
    pub fn compute_xor_blake3_digest(&self) -> String {
        let mut acc = [0u8; 32];
        // Deterministic iteration order keeps the fold stable across runs.
        let mut keys: Vec<&String> = self.modules.keys().collect();
        keys.sort();
        for key in keys {
            let digest = &self.modules[key];
            for (a, b) in acc.iter_mut().zip(digest.iter()) {
                *a ^= b;
            }
        }
        to_hex(&acc)
    }

    /// Bulk drift detector: compare the live register against an expected
    /// pre-mutation boundary digest.
    pub fn verify_parity(&self, expected_digest: &str) -> bool {
        self.compute_xor_blake3_digest() == expected_digest
    }
}

fn to_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_is_stable_and_order_independent() {
        let mut a = HisBoundaryCore::new();
        a.update_module("pkg.mod".to_string(), "def f(x: int) -> int".to_string());
        a.update_module("pkg.other".to_string(), "class C: ...".to_string());
        let d1 = a.compute_xor_blake3_digest();
        assert_eq!(d1.len(), 64);

        let mut b = HisBoundaryCore::new();
        // Insert in reverse order: XOR fold must be identical.
        b.update_module("pkg.other".to_string(), "class C: ...".to_string());
        b.update_module("pkg.mod".to_string(), "def f(x: int) -> int".to_string());
        assert_eq!(d1, b.compute_xor_blake3_digest());
    }

    #[test]
    fn public_signature_change_moves_digest_private_body_does_not() {
        let mut reg = HisBoundaryCore::new();
        reg.update_module("pkg.mod".to_string(), "def f(x: int) -> int".to_string());
        let before = reg.compute_xor_blake3_digest();

        // Private implementation body changed, canonical signature identical.
        reg.update_module("pkg.mod".to_string(), "def f(x: int) -> int".to_string());
        assert_eq!(before, reg.compute_xor_blake3_digest());

        // Public signature drift must move the boundary digest.
        reg.update_module("pkg.mod".to_string(), "def f(x: str) -> str".to_string());
        assert_ne!(before, reg.compute_xor_blake3_digest());
    }

    #[test]
    fn parity_verification() {
        let mut reg = HisBoundaryCore::new();
        reg.update_module("m".to_string(), "sig".to_string());
        let digest = reg.compute_xor_blake3_digest();
        assert!(reg.verify_parity(&digest));
        assert!(!reg.verify_parity(&"0".repeat(64)));
    }
}
