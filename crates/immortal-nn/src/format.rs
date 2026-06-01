//! `.sovereign` model format: constants, CRC32, loader, writer (test-only).
//!
//! Binary format. Single file. Self-describing. Deterministic byte layout.
//! Zero external dependencies (no protobuf, no flatbuffers, no serde).
//!
//! See the plan (Section 4) for the full layout specification.

use std::collections::{HashMap, HashSet};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::tensor::{Shape, Tensor};
use crate::{NnError, NnResult};

// ---------------------------------------------------------------------------
// Format constants
// ---------------------------------------------------------------------------

/// Magic bytes at the start of every `.sovereign` file. 10 bytes.
pub const MAGIC: &[u8; 10] = b"SOVEREIGN\0";

/// Current format version. Bump on any non-backward-compatible header change.
pub const VERSION: u32 = 1;

/// Fixed header size in bytes.
pub const HEADER_SIZE: usize = 64;

/// Fixed per-tensor index entry size in bytes.
pub const INDEX_ENTRY_SIZE: usize = 128;

/// Maximum tensor name length (UTF-8, null-padded inside the index entry).
pub const MAX_NAME_LEN: usize = 64;

/// Alignment requirement for each tensor's data blob (bytes).
pub const TENSOR_ALIGNMENT: usize = 64;

/// Dtype tag: f32 (the only dtype in v1).
pub const DTYPE_F32: u8 = 0;
/// Dtype tag: f16 (reserved, not yet supported).
pub const DTYPE_F16: u8 = 1;
/// Dtype tag: bf16 (reserved, not yet supported).
pub const DTYPE_BF16: u8 = 2;
/// Dtype tag: int8 (per-tensor affine quantized).
pub const DTYPE_INT8: u8 = 3;
/// Dtype tag: int4 (per-tensor affine quantized, 2 values packed per byte).
pub const DTYPE_INT4: u8 = 4;

// ---------------------------------------------------------------------------
// CRC32 (IEEE 802.3 polynomial 0xEDB88320), zero-dep
// ---------------------------------------------------------------------------

/// IEEE CRC32 polynomial (reflected): `0xEDB88320`.
const POLY: u32 = 0xEDB8_8320;

/// Compute the IEEE CRC32 of `bytes`.
#[must_use]
pub fn crc32(bytes: &[u8]) -> u32 {
    let table = crc32_table();
    let mut crc = 0xFFFF_FFFF_u32;
    for &b in bytes {
        let idx = ((crc ^ u32::from(b)) & 0xFF) as usize;
        crc = (crc >> 8) ^ table[idx];
    }
    crc ^ 0xFFFF_FFFF
}

/// Streaming CRC32 builder.
#[derive(Debug, Clone)]
pub struct Crc32 {
    state: u32,
}

impl Crc32 {
    /// Start a new CRC32 computation.
    #[must_use]
    pub fn new() -> Self {
        Self { state: 0xFFFF_FFFF }
    }

    /// Feed more bytes into the running CRC.
    pub fn update(&mut self, bytes: &[u8]) {
        let table = crc32_table();
        let mut crc = self.state;
        for &b in bytes {
            let idx = ((crc ^ u32::from(b)) & 0xFF) as usize;
            crc = (crc >> 8) ^ table[idx];
        }
        self.state = crc;
    }

    /// Finalize and return the CRC value.
    #[must_use]
    pub fn finalize(self) -> u32 {
        self.state ^ 0xFFFF_FFFF
    }
}

impl Default for Crc32 {
    fn default() -> Self {
        Self::new()
    }
}

fn crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut j = 0;
        while j < 8 {
            c = if c & 1 != 0 { (c >> 1) ^ POLY } else { c >> 1 };
            j += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
}

// ---------------------------------------------------------------------------
// Little-endian helpers (zero-dep)
// ---------------------------------------------------------------------------

fn read_u32_le(buf: &[u8]) -> u32 {
    u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]])
}

fn read_u64_le(buf: &[u8]) -> u64 {
    u64::from_le_bytes([
        buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
    ])
}

fn write_u32_le(buf: &mut [u8], v: u32) {
    buf[..4].copy_from_slice(&v.to_le_bytes());
}

fn write_u64_le(buf: &mut [u8], v: u64) {
    buf[..8].copy_from_slice(&v.to_le_bytes());
}

// ---------------------------------------------------------------------------
// Flat JSON parser (hand-rolled, zero-dep)
// ---------------------------------------------------------------------------

/// Parse a flat `{"key":"value", ...}` JSON string into a `HashMap`.
/// Only supports string keys and string values. No nesting, no arrays.
fn parse_flat_json(s: &str) -> NnResult<HashMap<String, String>> {
    let s = s.trim();
    if s.is_empty() || s == "{}" {
        return Ok(HashMap::new());
    }
    if !s.starts_with('{') || !s.ends_with('}') {
        return Err(NnError::Format(
            "metadata JSON must be a flat object".into(),
        ));
    }
    let inner = &s[1..s.len() - 1];
    let mut map = HashMap::new();
    let mut rest = inner.trim();
    while !rest.is_empty() {
        // Parse key
        let key = parse_json_string(&mut rest)?;
        rest = rest.trim_start();
        if !rest.starts_with(':') {
            return Err(NnError::Format("expected ':' after key".into()));
        }
        rest = rest[1..].trim_start();
        // Parse value
        let val = parse_json_string(&mut rest)?;
        rest = rest.trim_start();
        if rest.starts_with(',') {
            rest = rest[1..].trim_start();
        }
        map.insert(key, val);
    }
    Ok(map)
}

/// Parse a JSON string literal (with basic escape support) and advance `rest`.
fn parse_json_string(rest: &mut &str) -> NnResult<String> {
    let s = *rest;
    if !s.starts_with('"') {
        return Err(NnError::Format(format!(
            "expected '\"' at start of JSON string, got {:?}",
            s.chars().next()
        )));
    }
    let mut result = String::new();
    let mut chars = s[1..].chars();
    let mut consumed = 1; // opening quote
    loop {
        match chars.next() {
            None => return Err(NnError::Format("unterminated JSON string".into())),
            Some('"') => {
                consumed += 1;
                break;
            }
            Some('\\') => {
                consumed += 1;
                match chars.next() {
                    Some('"') => {
                        result.push('"');
                        consumed += 1;
                    }
                    Some('\\') => {
                        result.push('\\');
                        consumed += 1;
                    }
                    Some('/') => {
                        result.push('/');
                        consumed += 1;
                    }
                    Some('n') => {
                        result.push('\n');
                        consumed += 1;
                    }
                    Some('t') => {
                        result.push('\t');
                        consumed += 1;
                    }
                    Some('r') => {
                        result.push('\r');
                        consumed += 1;
                    }
                    Some(c) => {
                        result.push(c);
                        consumed += c.len_utf8();
                    }
                    None => return Err(NnError::Format("unterminated escape".into())),
                }
            }
            Some(c) => {
                result.push(c);
                consumed += c.len_utf8();
            }
        }
    }
    *rest = &s[consumed..];
    Ok(result)
}

/// Serialize a flat `HashMap<String, String>` as JSON.
fn serialize_flat_json(map: &HashMap<String, String>) -> String {
    if map.is_empty() {
        return "{}".to_string();
    }
    let mut pairs: Vec<_> = map.iter().collect();
    pairs.sort_by_key(|(k, _)| (*k).clone());
    let entries: Vec<String> = pairs
        .iter()
        .map(|(k, v)| format!("\"{}\":\"{}\"", escape_json(k), escape_json(v)))
        .collect();
    format!("{{{}}}", entries.join(","))
}

fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            _ => out.push(c),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tensor index entry (on-disk layout)
// ---------------------------------------------------------------------------

/// Parsed tensor index entry from the `.sovereign` file.
#[derive(Debug, Clone)]
struct TensorEntry {
    name: String,
    dtype: u8,
    ndim: u8,
    shape: [u32; 4],
    data_offset: u64,
    data_length: u64,
    checksum: u32,
}

impl TensorEntry {
    /// Parse one 128-byte index entry.
    fn from_bytes(buf: &[u8; INDEX_ENTRY_SIZE]) -> NnResult<Self> {
        // Name: first 64 bytes, UTF-8 null-padded
        let name_end = buf[..MAX_NAME_LEN]
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(MAX_NAME_LEN);
        let name = std::str::from_utf8(&buf[..name_end])
            .map_err(|e| NnError::Format(format!("tensor name not UTF-8: {e}")))?
            .to_string();
        if name.is_empty() {
            return Err(NnError::Format("empty tensor name".into()));
        }

        let dtype = buf[64];
        let ndim = buf[65];
        let shape = [
            read_u32_le(&buf[68..72]),
            read_u32_le(&buf[72..76]),
            read_u32_le(&buf[76..80]),
            read_u32_le(&buf[80..84]),
        ];
        let data_offset = read_u64_le(&buf[84..92]);
        let data_length = read_u64_le(&buf[92..100]);
        let checksum = read_u32_le(&buf[100..104]);

        Ok(Self {
            name,
            dtype,
            ndim,
            shape,
            data_offset,
            data_length,
            checksum,
        })
    }

    /// Serialize to a 128-byte index entry.
    fn to_bytes(&self) -> [u8; INDEX_ENTRY_SIZE] {
        let mut buf = [0u8; INDEX_ENTRY_SIZE];
        let name_bytes = self.name.as_bytes();
        let len = name_bytes.len().min(MAX_NAME_LEN);
        buf[..len].copy_from_slice(&name_bytes[..len]);
        buf[64] = self.dtype;
        buf[65] = self.ndim;
        // 66..68 reserved
        write_u32_le(&mut buf[68..72], self.shape[0]);
        write_u32_le(&mut buf[72..76], self.shape[1]);
        write_u32_le(&mut buf[76..80], self.shape[2]);
        write_u32_le(&mut buf[80..84], self.shape[3]);
        write_u64_le(&mut buf[84..92], self.data_offset);
        write_u64_le(&mut buf[92..100], self.data_length);
        write_u32_le(&mut buf[100..104], self.checksum);
        buf
    }
}

// ---------------------------------------------------------------------------
// SovereignModel
// ---------------------------------------------------------------------------

/// A loaded `.sovereign` model: named tensors + flat string metadata.
#[derive(Debug)]
pub struct SovereignModel {
    tensors: HashMap<String, Tensor>,
    metadata: HashMap<String, String>,
    /// Per-tensor expected CRC32 (for `validate_checksums`).
    checksums: HashMap<String, u32>,
}

impl SovereignModel {
    /// Load a `.sovereign` model from disk.
    ///
    /// Validates magic, version, tensor index, and metadata. Per-tensor CRC32
    /// checksums are stored but **not** validated during load — call
    /// [`Self::validate_checksums`] explicitly if needed.
    ///
    /// # Errors
    /// Returns [`NnError::Io`] on read failures, [`NnError::Format`] on
    /// corrupt headers / indices, [`NnError::UnknownArchitecture`] if the
    /// `architecture` metadata key names an unknown arch.
    pub fn load(path: &Path) -> NnResult<Self> {
        let mut f = std::fs::File::open(path)
            .map_err(|e| NnError::Io(format!("{}: {e}", path.display())))?;

        // -- Header (64 bytes) --
        let mut header = [0u8; HEADER_SIZE];
        f.read_exact(&mut header)
            .map_err(|e| NnError::Io(format!("header read: {e}")))?;

        // Magic
        if &header[..10] != MAGIC {
            return Err(NnError::Format("bad magic (not a .sovereign file)".into()));
        }
        // Version
        let version = read_u32_le(&header[10..14]);
        if version != VERSION {
            return Err(NnError::Format(format!(
                "unsupported version {version} (expected {VERSION})"
            )));
        }
        // Counts & offsets
        let num_tensors = read_u32_le(&header[14..18]) as usize;
        let metadata_offset = read_u64_le(&header[18..26]);
        let metadata_length = read_u64_le(&header[26..34]);
        // header[34..38] = file CRC32 (optional, 0 = unset) — skipped in v1
        // header[38..64] = reserved

        // -- Tensor index --
        let mut entries = Vec::with_capacity(num_tensors);
        for i in 0..num_tensors {
            let mut ebuf = [0u8; INDEX_ENTRY_SIZE];
            f.read_exact(&mut ebuf)
                .map_err(|e| NnError::Io(format!("index entry {i}: {e}")))?;
            entries.push(TensorEntry::from_bytes(&ebuf)?);
        }

        // Check name uniqueness
        let mut seen = HashSet::with_capacity(num_tensors);
        for entry in &entries {
            if !seen.insert(entry.name.clone()) {
                return Err(NnError::Format(format!(
                    "duplicate tensor name: {:?}",
                    entry.name
                )));
            }
        }

        // -- Tensor data --
        let mut tensors = HashMap::with_capacity(num_tensors);
        let mut checksums = HashMap::with_capacity(num_tensors);

        for entry in &entries {
            if entry.dtype != DTYPE_F32 {
                return Err(NnError::Format(format!(
                    "tensor {:?}: unsupported dtype {} (v1 supports f32 only)",
                    entry.name, entry.dtype
                )));
            }

            // Build Shape
            let dims = [
                entry.shape[0] as usize,
                entry.shape[1] as usize,
                entry.shape[2] as usize,
                entry.shape[3] as usize,
            ];
            let shape = Shape::new(dims, entry.ndim)?;

            let expected_bytes = shape.numel() * 4;
            if entry.data_length as usize != expected_bytes {
                return Err(NnError::Format(format!(
                    "tensor {:?}: data_length {} != expected {} (shape {:?})",
                    entry.name,
                    entry.data_length,
                    expected_bytes,
                    shape.dims()
                )));
            }

            f.seek(SeekFrom::Start(entry.data_offset))
                .map_err(|e| NnError::Io(format!("seek tensor {:?}: {e}", entry.name)))?;

            let mut raw = vec![0u8; expected_bytes];
            f.read_exact(&mut raw)
                .map_err(|e| NnError::Io(format!("read tensor {:?}: {e}", entry.name)))?;

            // Convert bytes to f32 (little-endian)
            let data: Vec<f32> = raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();

            let tensor = Tensor::from_vec(data, shape)?;
            checksums.insert(entry.name.clone(), entry.checksum);
            tensors.insert(entry.name.clone(), tensor);
        }

        // -- Metadata --
        let metadata = if metadata_length > 0 {
            f.seek(SeekFrom::Start(metadata_offset))
                .map_err(|e| NnError::Io(format!("seek metadata: {e}")))?;
            let mut mbuf = vec![0u8; metadata_length as usize];
            f.read_exact(&mut mbuf)
                .map_err(|e| NnError::Io(format!("read metadata: {e}")))?;
            let json_str = std::str::from_utf8(&mbuf)
                .map_err(|e| NnError::Format(format!("metadata not UTF-8: {e}")))?;
            parse_flat_json(json_str)?
        } else {
            HashMap::new()
        };

        Ok(Self {
            tensors,
            metadata,
            checksums,
        })
    }

    /// Get a tensor by name.
    ///
    /// # Errors
    /// Returns [`NnError::Format`] if the name is not found.
    pub fn tensor(&self, name: &str) -> NnResult<&Tensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| NnError::Format(format!("tensor not found: {name:?}")))
    }

    /// Get a metadata value by key.
    #[must_use]
    pub fn metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(String::as_str)
    }

    /// Shorthand for `metadata("architecture")`.
    ///
    /// Returns an empty sentinel when the key is absent so architecture
    /// dispatchers can report a typed [`NnError::UnknownArchitecture`]
    /// instead of panicking while loading untrusted model files.
    #[must_use]
    pub fn architecture(&self) -> &str {
        self.metadata.get("architecture").map_or("", String::as_str)
    }

    /// All tensor names in the model.
    #[must_use]
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(String::as_str).collect()
    }

    /// Number of tensors.
    #[must_use]
    pub fn num_tensors(&self) -> usize {
        self.tensors.len()
    }

    /// Validate CRC32 checksums for every tensor.
    ///
    /// # Errors
    /// Returns [`NnError::ChecksumMismatch`] on the first failing tensor.
    pub fn validate_checksums(&self) -> NnResult<()> {
        for (name, tensor) in &self.tensors {
            let expected = self.checksums.get(name).copied().unwrap_or(0);
            if expected == 0 {
                continue; // checksum not set
            }
            // Recompute CRC32 over the raw f32 LE bytes
            let raw: Vec<u8> = tensor.data().iter().flat_map(|v| v.to_le_bytes()).collect();
            let got = crc32(&raw);
            if got != expected {
                return Err(NnError::ChecksumMismatch {
                    name: name.clone(),
                    expected,
                    got,
                });
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Writer (used by tests and by immortal-nn-convert)
// ---------------------------------------------------------------------------

/// Builder for writing `.sovereign` files.
pub struct SovereignWriter {
    tensors: Vec<(String, Tensor)>,
    /// Raw tensor entries: (name, bytes, shape, dtype). Used for quantized data.
    raw_tensors: Vec<(String, Vec<u8>, Shape, u8)>,
    metadata: HashMap<String, String>,
}

impl SovereignWriter {
    /// Create a new writer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tensors: Vec::new(),
            raw_tensors: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add a named tensor (f32).
    pub fn add_tensor(&mut self, name: impl Into<String>, tensor: Tensor) {
        self.tensors.push((name.into(), tensor));
    }

    /// Add a raw tensor with explicit dtype (for quantized data).
    ///
    /// `data` is the raw byte payload. `shape` describes the logical element
    /// layout. `dtype` is one of `DTYPE_*` constants (e.g. `DTYPE_INT8`).
    pub fn add_raw_tensor(
        &mut self,
        name: impl Into<String>,
        data: Vec<u8>,
        shape: Shape,
        dtype: u8,
    ) {
        self.raw_tensors.push((name.into(), data, shape, dtype));
    }

    /// Set a metadata key.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Write the `.sovereign` file to `path`.
    ///
    /// # Errors
    /// Returns [`NnError::Io`] on write failures, [`NnError::Format`] if a
    /// tensor name exceeds [`MAX_NAME_LEN`].
    pub fn write_to(&self, path: &Path) -> NnResult<()> {
        let mut f = std::fs::File::create(path)
            .map_err(|e| NnError::Io(format!("{}: {e}", path.display())))?;

        let num_tensors = self.tensors.len() + self.raw_tensors.len();

        // Compute tensor data section start (after header + index)
        let index_end = HEADER_SIZE + num_tensors * INDEX_ENTRY_SIZE;
        let data_start = align_up(index_end, TENSOR_ALIGNMENT);

        // Pre-compute entries with offsets
        let mut entries: Vec<(TensorEntry, Vec<u8>)> = Vec::with_capacity(num_tensors);
        let mut offset = data_start as u64;

        // f32 tensors
        for (name, tensor) in &self.tensors {
            if name.len() > MAX_NAME_LEN {
                return Err(NnError::Format(format!(
                    "tensor name {name:?} exceeds {MAX_NAME_LEN} bytes"
                )));
            }
            let raw: Vec<u8> = tensor.data().iter().flat_map(|v| v.to_le_bytes()).collect();
            let checksum = crc32(&raw);
            let data_length = raw.len() as u64;

            let shape = tensor.shape();
            let mut dims = [1u32; 4];
            for (i, &d) in shape.dims().iter().enumerate() {
                dims[i] = d as u32;
            }

            entries.push((
                TensorEntry {
                    name: name.clone(),
                    dtype: DTYPE_F32,
                    ndim: shape.ndim(),
                    shape: dims,
                    data_offset: offset,
                    data_length,
                    checksum,
                },
                raw,
            ));

            offset = align_up((offset + data_length) as usize, TENSOR_ALIGNMENT) as u64;
        }

        // Raw tensors (quantized / non-f32)
        for (name, data, shape, dtype) in &self.raw_tensors {
            if name.len() > MAX_NAME_LEN {
                return Err(NnError::Format(format!(
                    "tensor name {name:?} exceeds {MAX_NAME_LEN} bytes"
                )));
            }
            let checksum = crc32(data);
            let data_length = data.len() as u64;

            let mut dims = [1u32; 4];
            for (i, &d) in shape.dims().iter().enumerate() {
                dims[i] = d as u32;
            }

            entries.push((
                TensorEntry {
                    name: name.clone(),
                    dtype: *dtype,
                    ndim: shape.ndim(),
                    shape: dims,
                    data_offset: offset,
                    data_length,
                    checksum,
                },
                data.clone(),
            ));

            offset = align_up((offset + data_length) as usize, TENSOR_ALIGNMENT) as u64;
        }

        // Metadata JSON
        let meta_json = serialize_flat_json(&self.metadata);
        let meta_bytes = meta_json.as_bytes();
        let metadata_offset = offset;
        let metadata_length = meta_bytes.len() as u64;

        // -- Write header (64 bytes) --
        let mut header = [0u8; HEADER_SIZE];
        header[..10].copy_from_slice(MAGIC);
        write_u32_le(&mut header[10..14], VERSION);
        write_u32_le(&mut header[14..18], num_tensors as u32);
        write_u64_le(&mut header[18..26], metadata_offset);
        write_u64_le(&mut header[26..34], metadata_length);
        f.write_all(&header)
            .map_err(|e| NnError::Io(format!("write header: {e}")))?;

        // -- Write index --
        for (entry, _) in &entries {
            f.write_all(&entry.to_bytes())
                .map_err(|e| NnError::Io(format!("write index: {e}")))?;
        }

        // -- Padding to data section --
        let current = HEADER_SIZE + num_tensors * INDEX_ENTRY_SIZE;
        if data_start > current {
            let pad = vec![0u8; data_start - current];
            f.write_all(&pad)
                .map_err(|e| NnError::Io(format!("write padding: {e}")))?;
        }

        // -- Write tensor data --
        for (i, (entry, raw)) in entries.iter().enumerate() {
            f.seek(SeekFrom::Start(entry.data_offset))
                .map_err(|e| NnError::Io(format!("seek tensor {i}: {e}")))?;
            f.write_all(raw)
                .map_err(|e| NnError::Io(format!("write tensor {i}: {e}")))?;
        }

        // -- Write metadata --
        f.seek(SeekFrom::Start(metadata_offset))
            .map_err(|e| NnError::Io(format!("seek metadata: {e}")))?;
        f.write_all(meta_bytes)
            .map_err(|e| NnError::Io(format!("write metadata: {e}")))?;

        Ok(())
    }
}

impl Default for SovereignWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Round `n` up to the next multiple of `align`.
fn align_up(n: usize, align: usize) -> usize {
    n.div_ceil(align) * align
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_empty_input() {
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn crc32_known_vector_123456789() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn crc32_streaming_matches_oneshot() {
        let data = b"the quick brown fox jumps over the lazy dog";
        let oneshot = crc32(data);
        let mut s = Crc32::new();
        s.update(&data[..10]);
        s.update(&data[10..25]);
        s.update(&data[25..]);
        assert_eq!(s.finalize(), oneshot);
    }

    #[test]
    fn magic_is_ten_bytes() {
        assert_eq!(MAGIC.len(), 10);
    }

    #[test]
    fn header_and_index_sizes_are_fixed() {
        assert_eq!(HEADER_SIZE, 64);
        assert_eq!(INDEX_ENTRY_SIZE, 128);
    }

    #[test]
    fn flat_json_roundtrip() {
        let mut map = HashMap::new();
        map.insert(
            "architecture".to_string(),
            "token_classifier_bilstm_crf".to_string(),
        );
        map.insert("num_heads".to_string(), "8".to_string());
        map.insert("with\"quote".to_string(), "val\\ue".to_string());
        let json = serialize_flat_json(&map);
        let parsed = parse_flat_json(&json).unwrap();
        assert_eq!(parsed, map);
    }

    #[test]
    fn flat_json_empty() {
        assert!(parse_flat_json("{}").unwrap().is_empty());
        assert!(parse_flat_json("").unwrap().is_empty());
    }

    #[test]
    fn flat_json_rejects_bad_input() {
        assert!(parse_flat_json("[1,2]").is_err());
    }

    #[test]
    fn sovereign_roundtrip() {
        let dir = std::env::temp_dir().join("immortal_nn_test_roundtrip");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_model.sovereign");

        // Build a small model
        let w1 = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::d2(2, 3)).unwrap();
        let w2 = Tensor::from_vec(
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            Shape::d2(2, 4),
        )
        .unwrap();

        let mut writer = SovereignWriter::new();
        writer.add_tensor("layer.weight", w1.clone());
        writer.add_tensor("layer.bias", w2.clone());
        writer.set_metadata("architecture", "token_classifier_bilstm_crf");
        writer.set_metadata("hidden_dim", "128");
        writer.write_to(&path).unwrap();

        // Load it back
        let model = SovereignModel::load(&path).unwrap();
        assert_eq!(model.num_tensors(), 2);
        assert_eq!(model.architecture(), "token_classifier_bilstm_crf");
        assert_eq!(model.metadata("hidden_dim"), Some("128"));

        let loaded_w1 = model.tensor("layer.weight").unwrap();
        assert_eq!(loaded_w1.data(), w1.data());
        assert_eq!(loaded_w1.shape().dims(), w1.shape().dims());

        let loaded_w2 = model.tensor("layer.bias").unwrap();
        assert_eq!(loaded_w2.data(), w2.data());

        // Checksums should validate
        model.validate_checksums().unwrap();

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn sovereign_roundtrip_3d_tensor() {
        let dir = std::env::temp_dir().join("immortal_nn_test_3d");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_3d.sovereign");

        let t = Tensor::from_vec((0..24).map(|x| x as f32).collect(), Shape::d3(2, 3, 4)).unwrap();

        let mut writer = SovereignWriter::new();
        writer.add_tensor("embedding", t.clone());
        writer.set_metadata("architecture", "sequence_classifier_transformer");
        writer.write_to(&path).unwrap();

        let model = SovereignModel::load(&path).unwrap();
        let loaded = model.tensor("embedding").unwrap();
        assert_eq!(loaded.data(), t.data());
        assert_eq!(loaded.shape().ndim(), 3);
        assert_eq!(loaded.shape().dims(), &[2, 3, 4]);
        model.validate_checksums().unwrap();

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn sovereign_bad_magic_rejected() {
        let dir = std::env::temp_dir().join("immortal_nn_test_bad_magic");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("bad_magic.sovereign");

        std::fs::write(&path, b"NOT_SOVEREIGN_AT_ALL_NOPE").unwrap();
        assert!(SovereignModel::load(&path).is_err());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn sovereign_checksum_mismatch_detected() {
        let dir = std::env::temp_dir().join("immortal_nn_test_cksum");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_cksum.sovereign");

        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::d2(2, 2)).unwrap();
        let mut writer = SovereignWriter::new();
        writer.add_tensor("w", t);
        writer.set_metadata("architecture", "biaffine_parser");
        writer.write_to(&path).unwrap();

        // Corrupt one byte in the tensor data region
        let mut raw = std::fs::read(&path).unwrap();
        // Tensor data starts after header(64) + 1 index entry(128) + alignment padding
        let data_off = align_up(64 + 128, 64);
        raw[data_off] ^= 0xFF;
        std::fs::write(&path, &raw).unwrap();

        let model = SovereignModel::load(&path).unwrap();
        assert!(model.validate_checksums().is_err());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn align_up_works() {
        assert_eq!(align_up(0, 64), 0);
        assert_eq!(align_up(1, 64), 64);
        assert_eq!(align_up(64, 64), 64);
        assert_eq!(align_up(65, 64), 128);
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            /// CRC32 is deterministic: same bytes always produce the same CRC.
            #[test]
            fn crc32_deterministic(
                data in proptest::collection::vec(0u8..=255, 0..=128),
            ) {
                let first  = crc32(&data);
                let second = crc32(&data);
                prop_assert_eq!(first, second, "CRC32 must be deterministic");
            }

            /// Streaming Crc32 with two splits equals single-pass crc32.
            #[test]
            fn crc32_streaming_equals_oneshot(
                data  in proptest::collection::vec(0u8..=255, 1..=64),
                split in 0usize..64,
            ) {
                let split = split % data.len();
                let oneshot = crc32(&data);
                let mut s = Crc32::new();
                s.update(&data[..split]);
                s.update(&data[split..]);
                prop_assert_eq!(s.finalize(), oneshot,
                    "streaming CRC32 must match one-shot for split at {}", split);
            }

            /// Different byte sequences produce different CRC values (collision rate
            /// is negligible — just one extra byte flipped is enough to distinguish).
            #[test]
            fn crc32_differs_on_distinct_data(
                data in proptest::collection::vec(0u8..=254, 1..=64),
            ) {
                // Flip the last byte to create a definitely different input
                let mut altered = data.clone();
                *altered.last_mut().unwrap() = altered.last().unwrap().wrapping_add(1);
                let c1 = crc32(&data);
                let c2 = crc32(&altered);
                prop_assert_ne!(c1, c2, "CRC32 collision on single-byte change");
            }

            /// flat JSON metadata roundtrip: set_metadata then load recovers the value.
            #[test]
            fn metadata_roundtrip(
                key   in "[a-z_]{1,12}",
                value in "[a-z0-9_]{1,20}",
            ) {
                let dir = std::env::temp_dir().join("immortal_nn_proptest_meta");
                std::fs::create_dir_all(&dir).unwrap();
                let path = dir.join(format!("meta_{}_{}.sovereign", key.len(), value.len()));

                let mut writer = SovereignWriter::new();
                // A minimal tensor is required for a valid file
                let t = Tensor::from_vec(vec![1.0f32], Shape::d2(1, 1)).unwrap();
                writer.add_tensor("dummy", t);
                writer.set_metadata(&key, &value);
                writer.set_metadata("architecture", "biaffine_parser");
                writer.write_to(&path).unwrap();

                let model = SovereignModel::load(&path).unwrap();
                let got = model.metadata(&key);
                prop_assert_eq!(got, Some(value.as_str()),
                    "metadata roundtrip failed for key {:?} / value {:?}", key, value);

                let _ = std::fs::remove_file(&path);
            }

            /// Shape written via SovereignWriter and loaded back is identical.
            #[test]
            fn shape_roundtrip_via_writer(
                d0 in 1usize..=4,
                d1 in 1usize..=4,
                d2 in 1usize..=4,
            ) {
                let dir = std::env::temp_dir().join("immortal_nn_proptest_shape");
                std::fs::create_dir_all(&dir).unwrap();
                let path = dir.join(format!("shape_{d0}_{d1}_{d2}.sovereign"));

                let shape = Shape::d3(d0, d1, d2);
                let n = d0 * d1 * d2;
                let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
                let t = Tensor::from_vec(data.clone(), shape).unwrap();

                let mut writer = SovereignWriter::new();
                writer.add_tensor("t", t);
                writer.set_metadata("architecture", "biaffine_parser");
                writer.write_to(&path).unwrap();

                let model = SovereignModel::load(&path).unwrap();
                let loaded = model.tensor("t").unwrap();
                prop_assert_eq!(loaded.shape().dims(), &[d0, d1, d2]);
                prop_assert_eq!(loaded.data(), data.as_slice());

                let _ = std::fs::remove_file(&path);
            }
        }
    }
}
