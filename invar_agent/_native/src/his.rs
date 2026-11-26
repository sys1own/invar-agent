//! 256-bit BLAKE3 Holographic Boundary Register (`his.rs`).
//!
//! Projects the bulk workspace state (every module's canonicalized public
//! interface signature) onto a compact, fixed-width 256-bit boundary vector
//! `H(d_Omega)` using a cryptographic Merkle homomorphism.  The register is
//! order-independent (XOR-folded) so that concurrent wavefront commits can be
//! merged without re-hashing the entire repository.

use std::collections::{HashMap, HashSet};
use std::fmt;

use serde::de::{self, Deserializer, MapAccess, SeqAccess, Visitor};

/// 1-byte domain separation prefixes for Invar-Agent vNext BLAKE3 commitments.
pub const DOMAIN_SCIIR: u8 = 0x01;
pub const DOMAIN_IHCG_V1: u8 = 0x02;
pub const DOMAIN_KERNEL_SOURCE: u8 = 0x03;
pub const DOMAIN_EVIDENCE_RECORD: u8 = 0x04;
pub const DOMAIN_VERDICT_RECEIPT: u8 = 0x05;

#[derive(Debug, Clone)]
enum JcsValue {
    Null,
    Bool(bool),
    Int(i64),
    Uint(u64),
    Float(f64),
    Str(String),
    Array(Vec<JcsValue>),
    Object(Vec<(String, JcsValue)>),
}

impl<'de> serde::Deserialize<'de> for JcsValue {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct JcsVisitor;
        impl<'de> Visitor<'de> for JcsVisitor {
            type Value = JcsValue;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a valid RFC 8785 JSON value without duplicate keys")
            }
            fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
                Ok(JcsValue::Null)
            }
            fn visit_bool<E: de::Error>(self, v: bool) -> Result<Self::Value, E> {
                Ok(JcsValue::Bool(v))
            }
            fn visit_i64<E: de::Error>(self, v: i64) -> Result<Self::Value, E> {
                Ok(JcsValue::Int(v))
            }
            fn visit_u64<E: de::Error>(self, v: u64) -> Result<Self::Value, E> {
                Ok(JcsValue::Uint(v))
            }
            fn visit_f64<E: de::Error>(self, v: f64) -> Result<Self::Value, E> {
                if !v.is_finite() {
                    return Err(E::custom("RFC 8785 forbids NaN/Infinity floats"));
                }
                Ok(JcsValue::Float(v))
            }
            fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
                Ok(JcsValue::Str(v.to_string()))
            }
            fn visit_string<E: de::Error>(self, v: String) -> Result<Self::Value, E> {
                Ok(JcsValue::Str(v))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut out = Vec::new();
                while let Some(elem) = seq.next_element()? {
                    out.push(elem);
                }
                Ok(JcsValue::Array(out))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut seen = HashSet::new();
                let mut entries = Vec::new();
                while let Some((k, v)) = map.next_entry::<String, JcsValue>()? {
                    if !seen.insert(k.clone()) {
                        return Err(de::Error::custom(format!(
                            "RFC 8785 duplicate object key rejected: {k}"
                        )));
                    }
                    entries.push((k, v));
                }
                Ok(JcsValue::Object(entries))
            }
        }
        deserializer.deserialize_any(JcsVisitor)
    }
}

/// Format an IEEE-754 finite f64 strictly per ECMA-262 Number::toString / RFC 8785 §3.2.2.3.
pub fn format_es2015_f64(v: f64) -> Result<String, String> {
    if !v.is_finite() {
        return Err("RFC 8785 forbids NaN or Infinity".into());
    }
    if v == 0.0 {
        return Ok("0".into());
    }
    let negative = v.is_sign_negative();
    let abs_v = v.abs();

    // Use serde_json's Ryū serializer for exact IEEE-754 shortest round-trip digits.
    let raw = serde_json::to_string(&abs_v).map_err(|e| e.to_string())?;
    let (mantissa_str, base_exp): (&str, i32) = if let Some((m, e)) = raw.split_once(['e', 'E']) {
        (
            m,
            e.parse::<i32>()
                .map_err(|_| format!("invalid exponent in {raw}"))?,
        )
    } else {
        (raw.as_str(), 0)
    };

    // Parse mantissa_str into normalized significant digits (no leading/trailing zeros)
    // and base-10 exponent `exp` such that `abs_v = d0.d1... * 10^exp`.
    let (int_part, frac_part) = mantissa_str.split_once('.').unwrap_or((mantissa_str, ""));
    let combined = format!("{int_part}{frac_part}");
    let first_nz = combined
        .find(|c: char| c != '0')
        .ok_or_else(|| format!("unexpected zero mantissa: {raw}"))?;
    let mut digits = combined[first_nz..].to_string();
    while digits.len() > 1 && digits.ends_with('0') {
        digits.pop();
    }
    let exp = base_exp + (int_part.len() as i32) - (first_nz as i32) - 1;

    let k = digits.len() as i32;
    let n = exp + 1; // abs_v = 0.<digits> * 10^n

    let mut out = String::with_capacity(28);
    if negative {
        out.push('-');
    }

    if k <= n && n <= 21 {
        out.push_str(&digits);
        for _ in 0..(n - k) {
            out.push('0');
        }
    } else if 0 < n && n <= 21 {
        let split = n as usize;
        out.push_str(&digits[..split]);
        out.push('.');
        out.push_str(&digits[split..]);
    } else if -6 < n && n <= 0 {
        out.push_str("0.");
        for _ in 0..(-n) {
            out.push('0');
        }
        out.push_str(&digits);
    } else {
        out.push(digits.as_bytes()[0] as char);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push('e');
        let e_val = n - 1;
        if e_val >= 0 {
            out.push('+');
            out.push_str(&e_val.to_string());
        } else {
            out.push('-');
            out.push_str(&(-e_val).to_string());
        }
    }

    Ok(out)
}

fn write_jcs_string(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{0008}' => out.push_str("\\b"),
            '\u{000C}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn write_jcs_value(val: &JcsValue, out: &mut String) -> Result<(), String> {
    match val {
        JcsValue::Null => out.push_str("null"),
        JcsValue::Bool(true) => out.push_str("true"),
        JcsValue::Bool(false) => out.push_str("false"),
        JcsValue::Int(i) => {
            // Integers within IEEE-754 exact range format as plain decimal; outside use f64 rule
            if (-9_007_199_254_740_991i64..=9_007_199_254_740_991i64).contains(i) {
                out.push_str(&i.to_string());
            } else {
                out.push_str(&format_es2015_f64(*i as f64)?);
            }
        }
        JcsValue::Uint(u) => {
            if *u <= 9_007_199_254_740_991u64 {
                out.push_str(&u.to_string());
            } else {
                out.push_str(&format_es2015_f64(*u as f64)?);
            }
        }
        JcsValue::Float(f) => {
            out.push_str(&format_es2015_f64(*f)?);
        }
        JcsValue::Str(s) => write_jcs_string(s, out),
        JcsValue::Array(items) => {
            out.push('[');
            for (idx, item) in items.iter().enumerate() {
                if idx > 0 {
                    out.push(',');
                }
                write_jcs_value(item, out)?;
            }
            out.push(']');
        }
        JcsValue::Object(entries) => {
            // RFC 8785 §3.2.3: sort object property names by UTF-16 code units.
            let mut sorted: Vec<&(String, JcsValue)> = entries.iter().collect();
            sorted.sort_by(|a, b| a.0.encode_utf16().cmp(b.0.encode_utf16()));
            out.push('{');
            for (idx, (k, v)) in sorted.iter().enumerate() {
                if idx > 0 {
                    out.push(',');
                }
                write_jcs_string(k, out);
                out.push(':');
                write_jcs_value(v, out)?;
            }
            out.push('}');
        }
    }
    Ok(())
}

/// Canonicalize a JSON string strictly according to RFC 8785 (JCS).
/// Rejects duplicate object keys, NaN/Infinity, and malformed JSON.
pub fn jcs_canonicalize_str(json_str: &str) -> Result<String, String> {
    let val: JcsValue = serde_json::from_str(json_str)
        .map_err(|e| format!("RFC 8785 JCS parse error: {e}"))?;
    let mut out = String::with_capacity(json_str.len());
    write_jcs_value(&val, &mut out)?;
    Ok(out)
}

/// Compute a 1-byte domain-separated BLAKE3 commitment: `BLAKE3(domain_byte || payload)`.
pub fn domain_byte_blake3_hex(domain_byte: u8, payload: &[u8]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&[domain_byte]);
    hasher.update(payload);
    hasher.finalize().to_hex().to_string()
}

/// Compute a 1-byte domain-separated BLAKE3 commitment over RFC 8785 JCS-canonicalized JSON.
pub fn commit_jcs_json(domain_byte: u8, json_str: &str) -> Result<String, String> {
    let canon = jcs_canonicalize_str(json_str)?;
    Ok(domain_byte_blake3_hex(domain_byte, canon.as_bytes()))
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SciIrSealReceipt {
    pub schema_version: String,
    pub sciir_commitment_blake3: String,
    pub ihcg_v1_commitment_blake3: String,
    pub canonical_sciir_jcs: String,
}

/// Single-authority Rust sealing entrypoint for `SciIR` (`invar.sciir.v2`).
/// Validates the JSON against `SciIR` (rejecting unknown fields & non-finite floats),
/// rejects duplicate keys, canonicalizes via RFC 8785 UTF-16 JCS, and computes
/// `0x01 || JCS(SciIR)` and `0x02 || JCS(IHCG_v1_projection)`.
pub fn seal_sciir_ihcg(sciir_json: &str) -> Result<SciIrSealReceipt, String> {
    // 1. Validate duplicate keys + RFC 8785 JCS on raw input
    let _raw_jcs = jcs_canonicalize_str(sciir_json)?;

    // 2. Deserialize strictly into SciIR and validate finiteness
    let ir: crate::sci_ir::SciIR = serde_json::from_str(sciir_json)
        .map_err(|e| format!("SciIR schema validation failed: {e}"))?;
    ir.validate()?;

    // 3. Serialize canonical SciIR struct -> RFC 8785 JCS -> 0x01 commitment
    let ir_serialized = serde_json::to_string(&ir).map_err(|e| e.to_string())?;
    let canonical_sciir_jcs = jcs_canonicalize_str(&ir_serialized)?;
    let sciir_commitment_blake3 =
        domain_byte_blake3_hex(DOMAIN_SCIIR, canonical_sciir_jcs.as_bytes());

    // 4. Deterministic projection to IHCG v1 summary object -> 0x02 commitment
    let ihcg_proj = serde_json::json!({
        "schema_version": "invar.ihcg.v1",
        "sciir_commitment_blake3": &sciir_commitment_blake3,
        "ir_id": &ir.ir_id,
        "archetype": &ir.archetype,
        "domain": &ir.domain,
        "symbol_count": ir.symbols.len(),
        "dynamics_count": ir.dynamics_or_residuals.len(),
        "observable_count": ir.observables.len(),
        "obligation_count": ir.obligations.len(),
    });
    let ihcg_proj_str = serde_json::to_string(&ihcg_proj).map_err(|e| e.to_string())?;
    let ihcg_v1_commitment_blake3 = commit_jcs_json(DOMAIN_IHCG_V1, &ihcg_proj_str)?;

    Ok(SciIrSealReceipt {
        schema_version: "invar.sciir.v2".to_string(),
        sciir_commitment_blake3,
        ihcg_v1_commitment_blake3,
        canonical_sciir_jcs,
    })
}

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

    #[test]
    fn rfc8785_jcs_utf16_sorting_and_numbers() {
        // UTF-16 code unit ordering: U+1F600 ("😀" -> 0xD83D 0xDE00) MUST sort BEFORE U+E000 (0xE000).
        let input = "{\"\\uE000\":2,\"😀\":1}";
        let canon = jcs_canonicalize_str(input).unwrap();
        assert_eq!(canon, "{\"😀\":1,\"\u{E000}\":2}");

        // RFC 8785 §3.2.2.3 & Appendix B number formatting vectors
        assert_eq!(format_es2015_f64(-0.0).unwrap(), "0");
        assert_eq!(format_es2015_f64(0.0).unwrap(), "0");
        assert_eq!(format_es2015_f64(1.0).unwrap(), "1");
        assert_eq!(format_es2015_f64(4.50).unwrap(), "4.5");
        assert_eq!(format_es2015_f64(2e-3).unwrap(), "0.002");
        assert_eq!(format_es2015_f64(1e-6).unwrap(), "0.000001");
        assert_eq!(format_es2015_f64(1e-7).unwrap(), "1e-7");
        assert_eq!(format_es2015_f64(1e20).unwrap(), "100000000000000000000");
        assert_eq!(format_es2015_f64(1e21).unwrap(), "1e+21");
        assert_eq!(format_es2015_f64(1e30).unwrap(), "1e+30");
        assert!(format_es2015_f64(f64::NAN).is_err());
        assert!(format_es2015_f64(f64::INFINITY).is_err());

        // Duplicate key rejection
        assert!(jcs_canonicalize_str(r#"{"a":1,"a":2}"#).is_err());

        // 1-byte domain separation
        let h1 = domain_byte_blake3_hex(DOMAIN_SCIIR, b"{}");
        let h2 = domain_byte_blake3_hex(DOMAIN_IHCG_V1, b"{}");
        let h3 = domain_byte_blake3_hex(DOMAIN_KERNEL_SOURCE, b"{}");
        let h4 = domain_byte_blake3_hex(DOMAIN_EVIDENCE_RECORD, b"{}");
        let h5 = domain_byte_blake3_hex(DOMAIN_VERDICT_RECEIPT, b"{}");
        assert_eq!(h1.len(), 64);
        assert_ne!(h1, h2);
        assert_ne!(h2, h3);
        assert_ne!(h3, h4);
        assert_ne!(h4, h5);
    }
}
