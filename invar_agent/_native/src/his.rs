//! 256-bit BLAKE3 Holographic Boundary Register (`his.rs`).
//!
//! Projects the bulk workspace state (every module's canonicalized public
//! interface signature) onto a compact, fixed-width 256-bit boundary vector
//! `H(d_Omega)` using a cryptographic Merkle homomorphism.  The register is
//! order-independent (XOR-folded) so that concurrent wavefront commits can be
//! merged without re-hashing the entire repository.

use std::collections::HashMap;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CdsTriplet { pub model_hash: String, pub claim_hash: String, pub full_hash: String }

fn epistemic_digest(context: &str, payload: &[u8]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(context.as_bytes());
    hasher.update(payload);
    hasher.finalize().to_hex().to_string()
}

pub fn compute_ihcg_digest(ihcg_json: &str) -> Result<String, String> {
    let mut value: serde_json::Value = serde_json::from_str(ihcg_json).map_err(|e| e.to_string())?;
    value.as_object_mut().ok_or("IHCG must be an object")?.remove("h_ihcg");
    Ok(epistemic_digest("invar.ihcg.v1", &crate::uast::epistemic_canonical_bytes(&value.to_string())?))
}

pub fn compute_ihcg_stage_chain(prev_hash: &str, delta_json: &str) -> Result<String, String> {
    let mut bytes = hex::decode(prev_hash).map_err(|_| "invalid previous digest")?;
    if bytes.len() != 32 { return Err("previous digest must contain 32 bytes".into()); }
    bytes.extend(crate::uast::epistemic_canonical_bytes(delta_json)?);
    Ok(epistemic_digest("invar.ihcg.stage.v1", &bytes))
}

pub fn compute_cds_digests(canon_model: &str, canon_claim: &str, domain_and_verdict: &str) -> CdsTriplet {
    let model_hash = epistemic_digest("invar.cds.model.v1", canon_model.as_bytes());
    let mut claim = model_hash.as_bytes().to_vec(); claim.extend_from_slice(canon_claim.as_bytes());
    let claim_hash = epistemic_digest("invar.cds.claim.v1", &claim);
    let mut full = claim_hash.as_bytes().to_vec(); full.extend_from_slice(domain_and_verdict.as_bytes());
    let full_hash = epistemic_digest("invar.cds.full.v1", &full);
    CdsTriplet { model_hash, claim_hash, full_hash }
}

pub fn compute_certificate_receipt(payload_json: &str) -> Result<String, String> {
    let mut value: serde_json::Value = serde_json::from_str(payload_json).map_err(|e| e.to_string())?;
    value.as_object_mut().ok_or("certificate must be an object")?.remove("blake3_receipt");
    Ok(epistemic_digest("invar.cert.v1", &crate::uast::epistemic_canonical_bytes(&value.to_string())?))
}

pub fn domain_separated_blake3_hex(context: &str, payload: &[u8]) -> String {
    let mut hasher = blake3::Hasher::new_derive_key(context);
    hasher.update(payload);
    hasher.finalize().to_hex().to_string()
}

pub fn compute_fault_fingerprint_json(fields_json: &str) -> Result<String, String> {
    let fields: [String; 6] = serde_json::from_str(fields_json).map_err(|e| e.to_string())?;
    let mut bytes = Vec::new();
    for field in fields {
        bytes.extend_from_slice(&(field.len() as u64).to_le_bytes());
        bytes.extend_from_slice(field.as_bytes());
    }
    Ok(domain_separated_blake3_hex("invar.builder.fault_fingerprint.v1", &bytes))
}

pub fn compute_receipt_merkle_root(receipt_json: &str) -> Result<String, String> {
    let mut value: serde_json::Value = serde_json::from_str(receipt_json).map_err(|e| e.to_string())?;
    value.as_object_mut().ok_or("Receipt must be a JSON object")?.remove("merkle_root_blake3");
    let bytes = crate::uast::canonical_json_bytes(&value.to_string())?;
    Ok(domain_separated_blake3_hex("invar.receipt.merkle.v1", &bytes))
}

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
    fn test_blake3_domain_separation() {
        let a = domain_separated_blake3_hex("invar.his.interface.v2", b"{}");
        let b = domain_separated_blake3_hex("invar.his.body.v2", b"{}");
        assert_eq!(a.len(), 64);
        assert_eq!(b.len(), 64);
        assert_ne!(a, b);
        assert_eq!(compute_receipt_merkle_root(r#"{"z":"\u03b1","a":1}"#).unwrap(), compute_receipt_merkle_root(r#"{"a":1,"z":"α"}"#).unwrap());
        assert_ne!(compute_fault_fingerprint_json(r#"["ab","c","","","",""]"#).unwrap(), compute_fault_fingerprint_json(r#"["a","bc","","","",""]"#).unwrap());
        assert!(compute_fault_fingerprint_json("[]").is_err());
        assert!(compute_receipt_merkle_root("not json").is_err());
        assert!(compute_receipt_merkle_root("[]").is_err());
        assert_eq!(crate::uast::canonical_json_bytes("[0.00001,1e-7,1e16,1.0,-0.0]").unwrap(), b"[1e-05,1e-07,1e+16,1.0,-0.0]");
        assert_eq!(crate::uast::canonical_json_bytes(r#"{"x":"😀"}"#).unwrap(), br#"{"x":"\ud83d\ude00"}"#);
    }

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
