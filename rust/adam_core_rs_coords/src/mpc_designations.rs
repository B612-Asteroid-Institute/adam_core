//! Strict codecs for Minor Planet Center packed designations.
//!
//! The focused functions intentionally accept one representation only: `pack_*`
//! accepts canonical human-readable text and `unpack_*` accepts canonical packed
//! text.  The generic dispatchers cover minor planets, comets/interstellar
//! objects, and natural satellites without treating unknown identifiers as MPC
//! designations.

use std::fmt;

const BASE62: &[u8; 62] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
const MAX_NUMBERED_MINOR_PLANET: u64 = 620_000 + 62_u64.pow(4) - 1;
const MAX_EXTENDED_SEQUENCE: u64 = 62_u64.pow(4) - 1;
const COMET_TYPES: &[u8] = b"PCDXAI";
const NUMBERED_COMET_TYPES: &[u8] = b"PDI";
const SATELLITE_PLANETS: &[u8] = b"MJSUNP";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MpcDesignationError {
    Value(String),
    Key(String),
    Index,
}

impl fmt::Display for MpcDesignationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Value(message) => f.write_str(message),
            Self::Key(key) => write!(f, "{key}"),
            Self::Index => f.write_str("string index out of range"),
        }
    }
}

impl std::error::Error for MpcDesignationError {}

type Result<T> = std::result::Result<T, MpcDesignationError>;

fn invalid(kind: &str, value: &str) -> MpcDesignationError {
    MpcDesignationError::Value(format!("invalid {kind}: {value}"))
}

fn validate_text(value: &str, kind: &str) -> Result<()> {
    if value.is_empty()
        || !value.is_ascii()
        || value.bytes().any(|byte| !(32..=126).contains(&byte))
        || value.starts_with(' ')
        || value.ends_with(' ')
        || value.contains("  ")
    {
        return Err(invalid(kind, value));
    }
    Ok(())
}

fn base62_value(byte: u8) -> Option<u64> {
    BASE62
        .iter()
        .position(|candidate| *candidate == byte)
        .map(|i| i as u64)
}

fn base62_digit(value: u64) -> Result<char> {
    BASE62
        .get(value as usize)
        .copied()
        .map(char::from)
        .ok_or_else(|| MpcDesignationError::Value(format!("base-62 value out of range: {value}")))
}

fn encode_base62(mut value: u64, width: usize) -> Result<String> {
    let mut bytes = vec![b'0'; width];
    for byte in bytes.iter_mut().rev() {
        *byte = *BASE62
            .get((value % 62) as usize)
            .ok_or_else(|| MpcDesignationError::Value("base-62 encoding failed".to_string()))?;
        value /= 62;
    }
    if value != 0 {
        return Err(MpcDesignationError::Value(
            "value does not fit in base-62 field".to_string(),
        ));
    }
    String::from_utf8(bytes)
        .map_err(|_| MpcDesignationError::Value("base-62 encoding failed".to_string()))
}

fn decode_base62(value: &str, kind: &str) -> Result<u64> {
    let mut decoded = 0_u64;
    for byte in value.bytes() {
        decoded = decoded
            .checked_mul(62)
            .and_then(|current| base62_value(byte).and_then(|part| current.checked_add(part)))
            .ok_or_else(|| invalid(kind, value))?;
    }
    Ok(decoded)
}

fn parse_canonical_u64(value: &str, kind: &str) -> Result<u64> {
    if value.is_empty()
        || !value.bytes().all(|byte| byte.is_ascii_digit())
        || (value.len() > 1 && value.starts_with('0'))
    {
        return Err(invalid(kind, value));
    }
    value.parse::<u64>().map_err(|_| invalid(kind, value))
}

fn is_half_month(byte: u8) -> bool {
    (b'A'..=b'Y').contains(&byte) && byte != b'I'
}

fn is_sequence_letter(byte: u8) -> bool {
    byte.is_ascii_uppercase() && byte != b'I'
}

fn sequence_letter_position(byte: u8) -> Option<u64> {
    if !is_sequence_letter(byte) {
        return None;
    }
    Some(if byte < b'I' {
        (byte - b'A') as u64
    } else {
        (byte - b'A' - 1) as u64
    })
}

fn sequence_letter(position: u64) -> Result<char> {
    if position >= 25 {
        return Err(MpcDesignationError::Value(format!(
            "provisional sequence-letter position out of range: {position}"
        )));
    }
    let offset = position as u8 + if position >= 8 { 1 } else { 0 };
    Ok(char::from(b'A' + offset))
}

fn encode_cycle(cycle: u64, allow_zero: bool, kind: &str) -> Result<String> {
    if cycle >= 620 || (!allow_zero && cycle == 0) {
        return Err(MpcDesignationError::Value(format!(
            "{kind} cycle out of range: {cycle}"
        )));
    }
    Ok(format!("{}{}", base62_digit(cycle / 10)?, cycle % 10))
}

fn decode_cycle(value: &str, allow_zero: bool, kind: &str) -> Result<u64> {
    if value.len() != 2 {
        return Err(invalid(kind, value));
    }
    let bytes = value.as_bytes();
    if !bytes[1].is_ascii_digit() {
        return Err(invalid(kind, value));
    }
    let tens = base62_value(bytes[0]).ok_or_else(|| invalid(kind, value))?;
    let cycle = tens * 10 + (bytes[1] - b'0') as u64;
    if cycle >= 620 || (!allow_zero && cycle == 0) {
        return Err(invalid(kind, value));
    }
    Ok(cycle)
}

fn century_code(year: i64, kind: &str) -> Result<char> {
    let century = year / 100;
    if !(10..=21).contains(&century) {
        return Err(MpcDesignationError::Value(format!(
            "{kind} year out of range: {year}"
        )));
    }
    Ok(char::from(b'A' + (century - 10) as u8))
}

fn decode_century(code: u8, year_in_century: u64, minimum: i64, kind: &str) -> Result<i64> {
    if !(b'A'..=b'L').contains(&code) || year_in_century > 99 {
        return Err(invalid(kind, &char::from(code).to_string()));
    }
    let year = (10 + (code - b'A') as i64) * 100 + year_in_century as i64;
    if year < minimum {
        return Err(MpcDesignationError::Value(format!(
            "{kind} year out of range: {year}"
        )));
    }
    Ok(year)
}

pub fn pack_numbered_designation(designation: &str) -> Result<String> {
    validate_text(designation, "numbered minor-planet designation")?;
    let number = parse_canonical_u64(designation, "numbered minor-planet designation")?;
    if !(1..=MAX_NUMBERED_MINOR_PLANET).contains(&number) {
        return Err(MpcDesignationError::Value(format!(
            "numbered minor-planet designation out of range: {designation}"
        )));
    }
    if number < 100_000 {
        return Ok(format!("{number:05}"));
    }
    if number < 620_000 {
        return Ok(format!(
            "{}{modulus:04}",
            base62_digit(number / 10_000)?,
            modulus = number % 10_000
        ));
    }
    Ok(format!("~{}", encode_base62(number - 620_000, 4)?))
}

pub fn unpack_numbered_designation(designation: &str) -> Result<String> {
    validate_text(designation, "packed numbered minor-planet designation")?;
    if designation.len() != 5 {
        return Err(invalid(
            "packed numbered minor-planet designation",
            designation,
        ));
    }
    let bytes = designation.as_bytes();
    let number = if bytes[0] == b'~' {
        620_000
            + decode_base62(
                &designation[1..],
                "packed numbered minor-planet designation",
            )?
    } else if bytes[0].is_ascii_digit() {
        if !bytes.iter().all(u8::is_ascii_digit) {
            return Err(invalid(
                "packed numbered minor-planet designation",
                designation,
            ));
        }
        designation
            .parse::<u64>()
            .map_err(|_| invalid("packed numbered minor-planet designation", designation))?
    } else {
        if base62_value(bytes[0])
            .filter(|value| (10..=61).contains(value))
            .is_none()
            || !bytes[1..].iter().all(u8::is_ascii_digit)
        {
            return Err(invalid(
                "packed numbered minor-planet designation",
                designation,
            ));
        }
        base62_value(bytes[0]).unwrap() * 10_000
            + designation[1..]
                .parse::<u64>()
                .map_err(|_| invalid("packed numbered minor-planet designation", designation))?
    };
    if !(1..=MAX_NUMBERED_MINOR_PLANET).contains(&number)
        || pack_numbered_designation(&number.to_string())? != designation
    {
        return Err(invalid(
            "packed numbered minor-planet designation",
            designation,
        ));
    }
    Ok(number.to_string())
}

fn survey_parts(designation: &str) -> Result<(u64, &'static str, &'static str)> {
    let (number_text, suffix) = designation
        .split_once(' ')
        .ok_or_else(|| invalid("survey designation", designation))?;
    let (packed_prefix, canonical_suffix) = match suffix {
        "P-L" => ("PLS", "P-L"),
        "T-1" => ("T1S", "T-1"),
        "T-2" => ("T2S", "T-2"),
        "T-3" => ("T3S", "T-3"),
        _ => return Err(invalid("survey designation", designation)),
    };
    let number = parse_canonical_u64(number_text, "survey designation")?;
    if !(1..=9999).contains(&number) {
        return Err(MpcDesignationError::Value(format!(
            "survey number out of range: {number}"
        )));
    }
    Ok((number, packed_prefix, canonical_suffix))
}

pub fn pack_survey_designation(designation: &str) -> Result<String> {
    validate_text(designation, "survey designation")?;
    let (number, prefix, _) = survey_parts(designation)?;
    Ok(format!("{prefix}{number:04}"))
}

pub fn unpack_survey_designation(designation: &str) -> Result<String> {
    validate_text(designation, "packed survey designation")?;
    if designation.len() != 7 || !designation[3..].bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(invalid("packed survey designation", designation));
    }
    let suffix = match &designation[..3] {
        "PLS" => "P-L",
        "T1S" => "T-1",
        "T2S" => "T-2",
        "T3S" => "T-3",
        _ => return Err(invalid("packed survey designation", designation)),
    };
    let number = designation[3..]
        .parse::<u64>()
        .map_err(|_| invalid("packed survey designation", designation))?;
    if !(1..=9999).contains(&number) {
        return Err(invalid("packed survey designation", designation));
    }
    Ok(format!("{number} {suffix}"))
}

#[derive(Debug)]
struct MinorProvisional {
    year: i64,
    half_month: u8,
    second_letter: u8,
    cycle: u64,
}

fn parse_minor_provisional(designation: &str) -> Result<MinorProvisional> {
    validate_text(designation, "provisional minor-planet designation")?;
    let (year_text, sequence) = designation
        .split_once(' ')
        .ok_or_else(|| invalid("provisional minor-planet designation", designation))?;
    if sequence.len() < 2 {
        return Err(invalid("provisional minor-planet designation", designation));
    }
    let sequence_bytes = sequence.as_bytes();
    if !is_half_month(sequence_bytes[0]) || !is_sequence_letter(sequence_bytes[1]) {
        return Err(invalid("provisional minor-planet designation", designation));
    }
    let cycle = if sequence.len() == 2 {
        0
    } else {
        parse_canonical_u64(&sequence[2..], "provisional minor-planet cycle")?
    };
    if sequence.len() > 2 && cycle == 0 {
        return Err(invalid("provisional minor-planet cycle", &sequence[2..]));
    }
    let year = if year_text.len() == 4 && year_text.starts_with('A') {
        if !year_text[1..].bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(invalid(
                "A-prefix provisional minor-planet designation",
                designation,
            ));
        }
        let year = 1000
            + year_text[1..].parse::<i64>().map_err(|_| {
                invalid("A-prefix provisional minor-planet designation", designation)
            })?;
        if !(1800..=1924).contains(&year) || cycle != 0 {
            return Err(invalid(
                "A-prefix provisional minor-planet designation",
                designation,
            ));
        }
        year
    } else {
        if year_text.len() != 4 || !year_text.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(invalid("provisional minor-planet designation", designation));
        }
        let year = year_text
            .parse::<i64>()
            .map_err(|_| invalid("provisional minor-planet designation", designation))?;
        if !(1925..=2199).contains(&year) {
            return Err(MpcDesignationError::Value(format!(
                "provisional minor-planet year out of range or requires A-prefix form: {year}"
            )));
        }
        year
    };
    Ok(MinorProvisional {
        year,
        half_month: sequence_bytes[0],
        second_letter: sequence_bytes[1],
        cycle,
    })
}

pub fn pack_provisional_designation(designation: &str) -> Result<String> {
    let parsed = parse_minor_provisional(designation)?;
    if parsed.cycle < 620 {
        let century = century_code(parsed.year, "provisional minor-planet")?;
        let cycle = encode_cycle(parsed.cycle, true, "provisional minor-planet")?;
        return Ok(format!(
            "{century}{:02}{}{}{}",
            parsed.year % 100,
            char::from(parsed.half_month),
            cycle,
            char::from(parsed.second_letter)
        ));
    }

    // The extended syntax stores year-2000 in one base-62 digit.  Its cited
    // field definition therefore covers 2000..=2061, not the former
    // uppercase-only 2010..=2035 assumption.
    if !(2000..=2061).contains(&parsed.year) {
        return Err(MpcDesignationError::Value(format!(
            "extended provisional minor-planet year out of range (2000-2061): {}",
            parsed.year
        )));
    }
    let position = sequence_letter_position(parsed.second_letter)
        .ok_or_else(|| invalid("provisional sequence letter", designation))?;
    let sequence = (parsed.cycle - 620)
        .checked_mul(25)
        .and_then(|value| value.checked_add(position))
        .ok_or_else(|| invalid("extended provisional minor-planet designation", designation))?;
    if sequence > MAX_EXTENDED_SEQUENCE {
        return Err(MpcDesignationError::Value(format!(
            "extended provisional sequence out of range: {sequence}"
        )));
    }
    Ok(format!(
        "_{}{}{}",
        base62_digit((parsed.year - 2000) as u64)?,
        char::from(parsed.half_month),
        encode_base62(sequence, 4)?
    ))
}

pub fn unpack_provisional_designation(designation: &str) -> Result<String> {
    validate_text(designation, "packed provisional minor-planet designation")?;
    if designation.len() != 7 {
        return Err(invalid(
            "packed provisional minor-planet designation",
            designation,
        ));
    }
    let bytes = designation.as_bytes();
    if bytes[0] == b'_' {
        let year_offset = base62_value(bytes[1]).ok_or_else(|| {
            invalid(
                "packed extended provisional minor-planet designation",
                designation,
            )
        })?;
        if !is_half_month(bytes[2]) {
            return Err(invalid(
                "packed extended provisional minor-planet designation",
                designation,
            ));
        }
        let sequence = decode_base62(
            &designation[3..],
            "packed extended provisional minor-planet designation",
        )?;
        let cycle = 620 + sequence / 25;
        let second = sequence_letter(sequence % 25)?;
        let year = 2000 + year_offset as i64;
        let result = format!("{year} {}{second}{cycle}", char::from(bytes[2]));
        if pack_provisional_designation(&result)? != designation {
            return Err(invalid(
                "packed extended provisional minor-planet designation",
                designation,
            ));
        }
        return Ok(result);
    }
    if !(b'I'..=b'L').contains(&bytes[0])
        || !bytes[1..3].iter().all(u8::is_ascii_digit)
        || !is_half_month(bytes[3])
        || !is_sequence_letter(bytes[6])
    {
        return Err(invalid(
            "packed provisional minor-planet designation",
            designation,
        ));
    }
    let year_in_century = designation[1..3]
        .parse::<u64>()
        .map_err(|_| invalid("packed provisional minor-planet designation", designation))?;
    let year = decode_century(bytes[0], year_in_century, 1800, "provisional minor-planet")?;
    let cycle = decode_cycle(
        &designation[4..6],
        true,
        "packed provisional minor-planet designation",
    )?;
    let prefix = if year < 1925 {
        format!("A{:03}", year - 1000)
    } else {
        year.to_string()
    };
    let suffix = if cycle == 0 {
        String::new()
    } else {
        cycle.to_string()
    };
    let result = format!(
        "{prefix} {}{}{suffix}",
        char::from(bytes[3]),
        char::from(bytes[6])
    );
    if pack_provisional_designation(&result)? != designation {
        return Err(invalid(
            "packed provisional minor-planet designation",
            designation,
        ));
    }
    Ok(result)
}

fn valid_fragment(fragment: &str) -> bool {
    (1..=2).contains(&fragment.len()) && fragment.bytes().all(is_sequence_letter)
}

fn parse_numbered_comet(designation: &str) -> Result<(u64, u8, Option<&str>, Option<&str>)> {
    if designation.is_empty() || designation.starts_with(' ') || designation.ends_with(' ') {
        return Err(invalid("numbered comet designation", designation));
    }
    let (identity, name) = match designation.split_once('/') {
        Some((identity, name))
            if !name.is_empty()
                && !name
                    .chars()
                    .any(|ch| matches!(ch, '|' | '\r' | '\n' | '\t'))
                && !name.starts_with(' ')
                && !name.ends_with(' ')
                && parse_signed_year(
                    name.split_once(' ').map_or(name, |(year, _)| year),
                    designation,
                )
                .is_err() =>
        {
            (identity, Some(name))
        }
        Some(_) => return Err(invalid("numbered comet designation", designation)),
        None => (designation, None),
    };
    validate_text(identity, "numbered comet designation")?;
    let (base, fragment) = match identity.split_once('-') {
        Some((base, fragment)) if valid_fragment(fragment) => (base, Some(fragment)),
        Some(_) => return Err(invalid("numbered comet designation", designation)),
        None => (identity, None),
    };
    if base.len() < 2 {
        return Err(invalid("numbered comet designation", designation));
    }
    let comet_type = *base.as_bytes().last().unwrap();
    if !NUMBERED_COMET_TYPES.contains(&comet_type) {
        return Err(invalid("numbered comet designation", designation));
    }
    let number = parse_canonical_u64(&base[..base.len() - 1], "numbered comet designation")?;
    if !(1..=9999).contains(&number) {
        return Err(MpcDesignationError::Value(format!(
            "numbered comet out of range: {number}"
        )));
    }
    Ok((number, comet_type, fragment, name))
}

pub fn pack_numbered_comet_designation(designation: &str) -> Result<String> {
    let (number, comet_type, fragment, _) = parse_numbered_comet(designation)?;
    Ok(format!(
        "{number:04}{}{}",
        char::from(comet_type),
        fragment.unwrap_or("").to_ascii_lowercase()
    ))
}

pub fn unpack_numbered_comet_designation(designation: &str) -> Result<String> {
    validate_text(designation, "packed numbered comet designation")?;
    if !(5..=7).contains(&designation.len())
        || !designation[..4].bytes().all(|byte| byte.is_ascii_digit())
        || !NUMBERED_COMET_TYPES.contains(&designation.as_bytes()[4])
        || !designation[5..]
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() && byte != b'i')
    {
        return Err(invalid("packed numbered comet designation", designation));
    }
    let number = designation[..4]
        .parse::<u64>()
        .map_err(|_| invalid("packed numbered comet designation", designation))?;
    if !(1..=9999).contains(&number) {
        return Err(invalid("packed numbered comet designation", designation));
    }
    let fragment = if designation.len() > 5 {
        format!("-{}", designation[5..].to_ascii_uppercase())
    } else {
        String::new()
    };
    let result = format!(
        "{number}{}{fragment}",
        char::from(designation.as_bytes()[4])
    );
    if pack_numbered_comet_designation(&result)? != designation {
        return Err(invalid("packed numbered comet designation", designation));
    }
    Ok(result)
}

#[derive(Debug)]
struct CometProvisional<'a> {
    number: Option<u64>,
    comet_type: u8,
    year: i64,
    half_month: u8,
    order: Option<u64>,
    asteroid_second: Option<u8>,
    asteroid_cycle: u64,
    fragment: Option<&'a str>,
}

fn parse_signed_year(value: &str, designation: &str) -> Result<i64> {
    if value == "0" || value == "-0" || value.starts_with('+') {
        return Err(invalid("comet provisional year", designation));
    }
    if let Some(rest) = value.strip_prefix('-') {
        let abs = parse_canonical_u64(rest, "comet provisional year")?;
        if !(1..=299).contains(&abs) {
            return Err(MpcDesignationError::Value(format!(
                "BCE comet year out of range: -{abs}"
            )));
        }
        return Ok(-(abs as i64));
    }
    let year = parse_canonical_u64(value, "comet provisional year")?;
    if !(1..=2199).contains(&year) {
        return Err(MpcDesignationError::Value(format!(
            "comet year out of range: {year}"
        )));
    }
    Ok(year as i64)
}

fn parse_comet_provisional(designation: &str) -> Result<CometProvisional<'_>> {
    validate_text(designation, "provisional comet designation")?;
    let (head, body) = designation
        .split_once('/')
        .ok_or_else(|| invalid("provisional comet designation", designation))?;
    let (number, comet_type) = if head.len() == 1 && COMET_TYPES.contains(&head.as_bytes()[0]) {
        (None, head.as_bytes()[0])
    } else {
        if head.len() < 2 {
            return Err(invalid("provisional comet designation", designation));
        }
        let comet_type = *head.as_bytes().last().unwrap();
        if !NUMBERED_COMET_TYPES.contains(&comet_type) {
            return Err(invalid(
                "numbered comet with provisional designation",
                designation,
            ));
        }
        let number = parse_canonical_u64(
            &head[..head.len() - 1],
            "numbered comet with provisional designation",
        )?;
        if !(1..=9999).contains(&number) {
            return Err(MpcDesignationError::Value(format!(
                "numbered comet out of range: {number}"
            )));
        }
        (Some(number), comet_type)
    };
    let (year_text, provisional) = body
        .split_once(' ')
        .ok_or_else(|| invalid("provisional comet designation", designation))?;
    let year = parse_signed_year(year_text, designation)?;
    if provisional.len() < 2 || !is_half_month(provisional.as_bytes()[0]) {
        return Err(invalid("provisional comet designation", designation));
    }
    let half_month = provisional.as_bytes()[0];
    let remainder = &provisional[1..];
    if remainder.as_bytes()[0].is_ascii_uppercase() {
        if year < 1925 || number.is_some() || !is_sequence_letter(remainder.as_bytes()[0]) {
            return Err(invalid("asteroid-style comet designation", designation));
        }
        let cycle = if remainder.len() == 1 {
            0
        } else {
            parse_canonical_u64(&remainder[1..], "asteroid-style comet cycle")?
        };
        if remainder.len() > 1 && cycle == 0 {
            return Err(invalid("asteroid-style comet cycle", designation));
        }
        if cycle >= 620 {
            return Err(MpcDesignationError::Value(
                "asteroid-style comet cycle cannot use extended minor-planet packing".to_string(),
            ));
        }
        return Ok(CometProvisional {
            number,
            comet_type,
            year,
            half_month,
            order: None,
            asteroid_second: Some(remainder.as_bytes()[0]),
            asteroid_cycle: cycle,
            fragment: None,
        });
    }
    let (order_text, fragment) = match remainder.split_once('-') {
        Some((order, fragment)) if valid_fragment(fragment) => (order, Some(fragment)),
        Some(_) => return Err(invalid("provisional comet fragment", designation)),
        None => (remainder, None),
    };
    let order = parse_canonical_u64(order_text, "provisional comet order")?;
    if !(1..=619).contains(&order) {
        return Err(MpcDesignationError::Value(format!(
            "provisional comet order out of range: {order}"
        )));
    }
    if number.is_some() && year < 1000 {
        return Err(MpcDesignationError::Value(
            "12-character numbered comet designations require years 1000-2199".to_string(),
        ));
    }
    if number.is_some() && fragment.is_some_and(|value| value.len() == 2) {
        return Err(MpcDesignationError::Value(
            "numbered comets with provisional designations do not define two-letter fragment packing"
                .to_string(),
        ));
    }
    Ok(CometProvisional {
        number,
        comet_type,
        year,
        half_month,
        order: Some(order),
        asteroid_second: None,
        asteroid_cycle: 0,
        fragment,
    })
}

fn encode_bce_year(year: i64) -> Result<(char, u64)> {
    let absolute = -year;
    let (prefix, century) = match absolute {
        1..=99 => ('/', 0),
        100..=199 => ('.', 100),
        200..=299 => ('-', 200),
        _ => {
            return Err(MpcDesignationError::Value(format!(
                "BCE comet year out of range: {year}"
            )))
        }
    };
    Ok((prefix, (99 - (absolute - century)) as u64))
}

fn decode_bce_year(prefix: u8, code: u64) -> Result<i64> {
    if code > 99 {
        return Err(MpcDesignationError::Value(
            "invalid BCE comet year code".to_string(),
        ));
    }
    let century = match prefix {
        b'/' => 0,
        b'.' => 100,
        b'-' => 200,
        _ => {
            return Err(MpcDesignationError::Value(
                "invalid BCE comet year prefix".to_string(),
            ))
        }
    };
    let absolute = century + 99 - code as i64;
    if absolute == 0 {
        return Err(MpcDesignationError::Value(
            "BCE comet year zero is invalid".to_string(),
        ));
    }
    Ok(-absolute)
}

fn pack_comet_provisional_parsed(parsed: &CometProvisional<'_>) -> Result<String> {
    let component = if let Some(second) = parsed.asteroid_second {
        let minor = format!(
            "{} {}{}{}",
            parsed.year,
            char::from(parsed.half_month),
            char::from(second),
            if parsed.asteroid_cycle == 0 {
                String::new()
            } else {
                parsed.asteroid_cycle.to_string()
            }
        );
        pack_provisional_designation(&minor)?
    } else {
        let order = parsed.order.unwrap();
        let cycle = encode_cycle(order, false, "provisional comet")?;
        let fragment = parsed.fragment.unwrap_or("0").to_ascii_lowercase();
        if parsed.year >= 1000 {
            format!(
                "{}{:02}{}{}{}",
                century_code(parsed.year, "provisional comet")?,
                parsed.year % 100,
                char::from(parsed.half_month),
                cycle,
                fragment
            )
        } else if parsed.year > 0 {
            format!(
                "{:03}{}{}{}",
                parsed.year,
                char::from(parsed.half_month),
                cycle,
                fragment
            )
        } else {
            let (prefix, code) = encode_bce_year(parsed.year)?;
            format!(
                "{prefix}{code:02}{}{}{}",
                char::from(parsed.half_month),
                cycle,
                fragment
            )
        }
    };
    if let Some(number) = parsed.number {
        Ok(format!(
            "{number:04}{}{}",
            char::from(parsed.comet_type),
            component
        ))
    } else {
        Ok(format!("{}{}", char::from(parsed.comet_type), component))
    }
}

pub fn pack_provisional_comet_designation(designation: &str) -> Result<String> {
    let parsed = parse_comet_provisional(designation)?;
    if parsed.fragment.is_some_and(|value| value.len() == 2) {
        // MPC has no packed policy for a two-letter provisional-comet fragment.
        // Preserve the documented canonical unpacked form rather than inventing one.
        return Ok(designation.to_string());
    }
    pack_comet_provisional_parsed(&parsed)
}

fn decode_comet_component(component: &str, comet_type: u8, number: Option<u64>) -> Result<String> {
    if component.len() != 7 {
        return Err(invalid("packed provisional comet component", component));
    }
    let bytes = component.as_bytes();
    let asteroid_style = (b'I'..=b'L').contains(&bytes[0]) && is_sequence_letter(bytes[6]);
    if asteroid_style {
        if number.is_some() {
            return Err(invalid(
                "packed numbered asteroid-style comet designation",
                component,
            ));
        }
        let provisional = unpack_provisional_designation(component)?;
        return Ok(format!("{}/{}", char::from(comet_type), provisional));
    }
    if !(b'A'..=b'L').contains(&bytes[0])
        || !bytes[1..3].iter().all(u8::is_ascii_digit)
        || !is_half_month(bytes[3])
        || !(bytes[6] == b'0' || (bytes[6].is_ascii_lowercase() && bytes[6] != b'i'))
    {
        return Err(invalid("packed provisional comet component", component));
    }
    let yy = component[1..3]
        .parse::<u64>()
        .map_err(|_| invalid("packed provisional comet component", component))?;
    let year = decode_century(bytes[0], yy, 1000, "provisional comet")?;
    let order = decode_cycle(
        &component[4..6],
        false,
        "packed provisional comet component",
    )?;
    let fragment = if bytes[6] == b'0' {
        String::new()
    } else {
        format!("-{}", char::from(bytes[6].to_ascii_uppercase()))
    };
    let head = number.map_or_else(
        || char::from(comet_type).to_string(),
        |number| format!("{number}{}", char::from(comet_type)),
    );
    Ok(format!(
        "{head}/{year} {}{order}{fragment}",
        char::from(bytes[3])
    ))
}

fn unpack_ancient_comet(designation: &str) -> Result<String> {
    let bytes = designation.as_bytes();
    if designation.len() != 8 || !COMET_TYPES.contains(&bytes[0]) {
        return Err(invalid("packed ancient comet designation", designation));
    }
    let (year, half_index, cycle_start) = if bytes[1].is_ascii_digit() {
        if !bytes[1..4].iter().all(u8::is_ascii_digit) {
            return Err(invalid("packed ancient comet designation", designation));
        }
        let year = designation[1..4]
            .parse::<i64>()
            .map_err(|_| invalid("packed ancient comet designation", designation))?;
        if !(1..=999).contains(&year) {
            return Err(invalid("packed ancient comet designation", designation));
        }
        (year, 4, 5)
    } else if b"/.-".contains(&bytes[1]) && bytes[2..4].iter().all(u8::is_ascii_digit) {
        let code = designation[2..4]
            .parse::<u64>()
            .map_err(|_| invalid("packed BCE comet designation", designation))?;
        (decode_bce_year(bytes[1], code)?, 4, 5)
    } else {
        return Err(invalid("packed ancient comet designation", designation));
    };
    if !is_half_month(bytes[half_index])
        || !(bytes[7] == b'0' || (bytes[7].is_ascii_lowercase() && bytes[7] != b'i'))
    {
        return Err(invalid("packed ancient comet designation", designation));
    }
    let order = decode_cycle(
        &designation[cycle_start..cycle_start + 2],
        false,
        "packed ancient comet designation",
    )?;
    let fragment = if bytes[7] == b'0' {
        String::new()
    } else {
        format!("-{}", char::from(bytes[7].to_ascii_uppercase()))
    };
    Ok(format!(
        "{}/{} {}{order}{fragment}",
        char::from(bytes[0]),
        year,
        char::from(bytes[half_index])
    ))
}

pub fn unpack_provisional_comet_designation(designation: &str) -> Result<String> {
    validate_text(designation, "packed provisional comet designation")?;
    if designation.contains('/') && designation.contains(' ') {
        let parsed = parse_comet_provisional(designation)?;
        if parsed.number.is_none() && parsed.fragment.is_some_and(|value| value.len() == 2) {
            return Ok(designation.to_string());
        }
        return Err(invalid("packed provisional comet designation", designation));
    }
    let result = if designation.len() == 12 {
        if !designation[..4].bytes().all(|byte| byte.is_ascii_digit())
            || !NUMBERED_COMET_TYPES.contains(&designation.as_bytes()[4])
        {
            return Err(invalid(
                "packed numbered comet with provisional designation",
                designation,
            ));
        }
        let number = designation[..4].parse::<u64>().map_err(|_| {
            invalid(
                "packed numbered comet with provisional designation",
                designation,
            )
        })?;
        if !(1..=9999).contains(&number) {
            return Err(invalid(
                "packed numbered comet with provisional designation",
                designation,
            ));
        }
        decode_comet_component(&designation[5..], designation.as_bytes()[4], Some(number))?
    } else if designation.len() == 8 {
        if !COMET_TYPES.contains(&designation.as_bytes()[0]) {
            return Err(invalid("packed provisional comet designation", designation));
        }
        if designation.as_bytes()[1].is_ascii_digit() || b"/.-".contains(&designation.as_bytes()[1])
        {
            unpack_ancient_comet(designation)?
        } else {
            decode_comet_component(&designation[1..], designation.as_bytes()[0], None)?
        }
    } else {
        return Err(invalid("packed provisional comet designation", designation));
    };
    if pack_provisional_comet_designation(&result)? != designation {
        return Err(invalid(
            "non-canonical packed provisional comet designation",
            designation,
        ));
    }
    Ok(result)
}

pub fn pack_comet_designation(designation: &str) -> Result<String> {
    if parse_comet_provisional(designation).is_ok() {
        pack_provisional_comet_designation(designation)
    } else {
        pack_numbered_comet_designation(designation)
    }
}

pub fn unpack_comet_designation(designation: &str) -> Result<String> {
    if (designation.contains('/') && designation.contains(' '))
        || matches!(designation.len(), 8 | 12)
    {
        unpack_provisional_comet_designation(designation)
    } else {
        unpack_numbered_comet_designation(designation)
    }
}

fn satellite_planet_name(code: u8) -> Option<&'static str> {
    match code {
        b'M' => Some("Mars"),
        b'J' => Some("Jupiter"),
        b'S' => Some("Saturn"),
        b'U' => Some("Uranus"),
        b'N' => Some("Neptune"),
        b'P' => Some("Pluto"),
        _ => None,
    }
}

fn satellite_planet_code(name: &str) -> Option<u8> {
    match name {
        "Mars" => Some(b'M'),
        "Jupiter" => Some(b'J'),
        "Saturn" => Some(b'S'),
        "Uranus" => Some(b'U'),
        "Neptune" => Some(b'N'),
        "Pluto" => Some(b'P'),
        _ => None,
    }
}

fn roman_value(roman: &str) -> Option<u64> {
    if roman.is_empty() || !roman.bytes().all(|byte| b"IVXLCDM".contains(&byte)) {
        return None;
    }
    let digit = |byte| match byte {
        b'I' => 1,
        b'V' => 5,
        b'X' => 10,
        b'L' => 50,
        b'C' => 100,
        b'D' => 500,
        b'M' => 1000,
        _ => 0,
    };
    let mut total = 0_i64;
    let mut previous = 0_i64;
    for byte in roman.bytes().rev() {
        let value = digit(byte);
        if value < previous {
            total = total.checked_sub(value)?;
        } else {
            total = total.checked_add(value)?;
            previous = value;
        }
    }
    let total = u64::try_from(total).ok()?;
    if !(1..=999).contains(&total) || to_roman(total) != roman {
        return None;
    }
    Some(total)
}

fn to_roman(mut value: u64) -> String {
    let symbols = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ];
    let mut result = String::new();
    for (number, symbol) in symbols {
        while value >= number {
            result.push_str(symbol);
            value -= number;
        }
    }
    result
}

pub fn pack_permanent_satellite_designation(designation: &str) -> Result<String> {
    validate_text(designation, "permanent natural-satellite designation")?;
    let (planet_name, roman) = designation
        .split_once(' ')
        .ok_or_else(|| invalid("permanent natural-satellite designation", designation))?;
    let code = satellite_planet_code(planet_name)
        .ok_or_else(|| invalid("permanent natural-satellite planet", planet_name))?;
    let number = roman_value(roman)
        .ok_or_else(|| invalid("permanent natural-satellite Roman numeral", roman))?;
    Ok(format!("{}{number:03}S", char::from(code)))
}

pub fn unpack_permanent_satellite_designation(designation: &str) -> Result<String> {
    validate_text(
        designation,
        "packed permanent natural-satellite designation",
    )?;
    let bytes = designation.as_bytes();
    if designation.len() != 5
        || !SATELLITE_PLANETS.contains(&bytes[0])
        || !bytes[1..4].iter().all(u8::is_ascii_digit)
        || bytes[4] != b'S'
    {
        return Err(invalid(
            "packed permanent natural-satellite designation",
            designation,
        ));
    }
    let number = designation[1..4].parse::<u64>().map_err(|_| {
        invalid(
            "packed permanent natural-satellite designation",
            designation,
        )
    })?;
    if !(1..=999).contains(&number) {
        return Err(invalid(
            "packed permanent natural-satellite designation",
            designation,
        ));
    }
    Ok(format!(
        "{} {}",
        satellite_planet_name(bytes[0]).unwrap(),
        to_roman(number)
    ))
}

pub fn pack_provisional_satellite_designation(designation: &str) -> Result<String> {
    validate_text(designation, "provisional natural-satellite designation")?;
    let body = designation
        .strip_prefix("S/")
        .ok_or_else(|| invalid("provisional natural-satellite designation", designation))?;
    let parts: Vec<&str> = body.split(' ').collect();
    if parts.len() != 3 || parts[0].len() != 4 || parts[1].len() != 1 {
        return Err(invalid(
            "provisional natural-satellite designation",
            designation,
        ));
    }
    let year = parse_canonical_u64(parts[0], "provisional natural-satellite year")? as i64;
    if !(1800..=2199).contains(&year) || !SATELLITE_PLANETS.contains(&parts[1].as_bytes()[0]) {
        return Err(invalid(
            "provisional natural-satellite designation",
            designation,
        ));
    }
    let number = parse_canonical_u64(parts[2], "provisional natural-satellite number")?;
    let cycle = encode_cycle(number, false, "provisional natural-satellite")?;
    Ok(format!(
        "S{}{:02}{}{}0",
        century_code(year, "provisional natural-satellite")?,
        year % 100,
        parts[1],
        cycle
    ))
}

pub fn unpack_provisional_satellite_designation(designation: &str) -> Result<String> {
    validate_text(
        designation,
        "packed provisional natural-satellite designation",
    )?;
    let bytes = designation.as_bytes();
    if designation.len() != 8
        || bytes[0] != b'S'
        || !(b'I'..=b'L').contains(&bytes[1])
        || !bytes[2..4].iter().all(u8::is_ascii_digit)
        || !SATELLITE_PLANETS.contains(&bytes[4])
        || bytes[7] != b'0'
    {
        return Err(invalid(
            "packed provisional natural-satellite designation",
            designation,
        ));
    }
    let yy = designation[2..4].parse::<u64>().map_err(|_| {
        invalid(
            "packed provisional natural-satellite designation",
            designation,
        )
    })?;
    let year = decode_century(bytes[1], yy, 1800, "provisional natural-satellite")?;
    let number = decode_cycle(
        &designation[5..7],
        false,
        "packed provisional natural-satellite designation",
    )?;
    let result = format!("S/{year} {} {number}", char::from(bytes[4]));
    if pack_provisional_satellite_designation(&result)? != designation {
        return Err(invalid(
            "packed provisional natural-satellite designation",
            designation,
        ));
    }
    Ok(result)
}

pub fn pack_satellite_designation(designation: &str) -> Result<String> {
    if designation.starts_with("S/") {
        pack_provisional_satellite_designation(designation)
    } else {
        pack_permanent_satellite_designation(designation)
    }
}

pub fn unpack_satellite_designation(designation: &str) -> Result<String> {
    if designation.len() == 8 {
        unpack_provisional_satellite_designation(designation)
    } else {
        unpack_permanent_satellite_designation(designation)
    }
}

pub fn pack_mpc_designation(designation: &str) -> Result<String> {
    if designation.is_empty() || designation.starts_with(' ') || designation.ends_with(' ') {
        return Err(invalid("MPC designation", designation));
    }
    if designation.bytes().all(|byte| byte.is_ascii_digit()) {
        return pack_numbered_designation(designation);
    }
    if designation.starts_with("S/")
        || satellite_planet_code(designation.split_once(' ').map_or("", |part| part.0)).is_some()
    {
        return pack_satellite_designation(designation);
    }
    if designation.contains('/') && designation.contains(' ') {
        return pack_comet_designation(designation);
    }
    if parse_numbered_comet(designation).is_ok() {
        return pack_numbered_comet_designation(designation);
    }
    if survey_parts(designation).is_ok() {
        return pack_survey_designation(designation);
    }
    if is_canonical_minor_provisional(designation) {
        return pack_provisional_designation(designation);
    }
    Err(invalid("unpacked MPC designation", designation))
}

pub fn unpack_mpc_designation(designation: &str) -> Result<String> {
    validate_text(designation, "packed MPC designation")?;
    if designation.contains('/') && designation.contains(' ') {
        return unpack_provisional_comet_designation(designation);
    }
    match designation.len() {
        12 => unpack_provisional_comet_designation(designation),
        8 => {
            if designation.starts_with('S') {
                unpack_provisional_satellite_designation(designation)
            } else {
                unpack_provisional_comet_designation(designation)
            }
        }
        7 => {
            if designation.starts_with("PLS")
                || designation.starts_with("T1S")
                || designation.starts_with("T2S")
                || designation.starts_with("T3S")
            {
                unpack_survey_designation(designation)
            } else if unpack_numbered_comet_designation(designation).is_ok() {
                unpack_numbered_comet_designation(designation)
            } else {
                unpack_provisional_designation(designation)
            }
        }
        6 => unpack_numbered_comet_designation(designation),
        5 => {
            if unpack_numbered_comet_designation(designation).is_ok() {
                unpack_numbered_comet_designation(designation)
            } else if unpack_permanent_satellite_designation(designation).is_ok() {
                unpack_permanent_satellite_designation(designation)
            } else {
                unpack_numbered_designation(designation)
            }
        }
        _ => Err(invalid("packed MPC designation", designation)),
    }
}

fn is_canonical_minor_provisional(value: &str) -> bool {
    let Ok(packed) = pack_provisional_designation(value) else {
        return false;
    };
    unpack_provisional_designation(&packed).is_ok_and(|unpacked| unpacked == value)
}

fn looks_like_malformed_minor_identity(number_text: &str, display_tail: &str) -> bool {
    if number_text.len() != 4
        || !number_text.bytes().all(|byte| byte.is_ascii_digit())
        || !display_tail
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b' ' | b'-'))
    {
        return false;
    }

    let upper_tail = display_tail.to_ascii_uppercase();
    let compact_tail = upper_tail.replace(' ', "");
    let normalized = format!("{number_text} {compact_tail}");
    if survey_parts(&normalized).is_ok() {
        return true;
    }

    let Ok(year) = number_text.parse::<u64>() else {
        return false;
    };
    if !(1800..=2199).contains(&year) {
        return false;
    }
    if is_canonical_minor_provisional(&normalized) {
        return true;
    }

    let bytes = compact_tail.as_bytes();
    if bytes.is_empty() || !is_half_month(bytes[0]) {
        return false;
    }
    if bytes.len() == 1 {
        return true;
    }
    if !is_sequence_letter(bytes[1]) {
        return false;
    }

    // Preserve title-cased names of four or more characters while rejecting
    // bounded tails that retain a genuine provisional-designation prefix.
    compact_tail.len() <= 3
        || (compact_tail.len() <= 4 && !display_tail.bytes().any(|byte| byte.is_ascii_lowercase()))
}

fn is_canonical_proper_name(value: &str) -> bool {
    !value.is_empty()
        && value.chars().next().is_some_and(char::is_uppercase)
        && !value.chars().any(|ch| {
            ch.is_ascii_digit()
                || ch.is_control()
                || matches!(ch, '|' | '(' | ')' | '\t' | '\r' | '\n')
        })
        && !value.starts_with(' ')
        && !value.ends_with(' ')
        && !value.contains("  ")
}

fn is_tracking_identifier(value: &str) -> bool {
    (1..=8).contains(&value.len())
        && value.as_bytes()[0].is_ascii_alphabetic()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
}

fn is_ades_provisional_satellite(value: &str) -> bool {
    let Ok(()) = validate_text(value, "provisional natural-satellite designation") else {
        return false;
    };
    let Some(body) = value.strip_prefix("S/") else {
        return false;
    };
    let Some((year_text, remainder)) = body.split_once(' ') else {
        return false;
    };
    let Ok(year) = parse_canonical_u64(year_text, "provisional natural-satellite year") else {
        return false;
    };
    if year_text.len() != 4 || !(1800..=2199).contains(&year) {
        return false;
    }
    let Some((primary, satellite_text)) = remainder.rsplit_once(' ') else {
        return false;
    };
    let Ok(satellite) = parse_canonical_u64(satellite_text, "provisional natural-satellite number")
    else {
        return false;
    };
    if satellite == 0 {
        return false;
    }
    if primary.len() == 1 && SATELLITE_PLANETS.contains(&primary.as_bytes()[0]) {
        let Ok(packed) = pack_provisional_satellite_designation(value) else {
            return false;
        };
        return unpack_provisional_satellite_designation(&packed)
            .is_ok_and(|unpacked| unpacked == value);
    }
    let Some(inner) = primary
        .strip_prefix('(')
        .and_then(|candidate| candidate.strip_suffix(')'))
    else {
        return false;
    };
    parse_canonical_u64(inner, "satellite primary")
        .is_ok_and(|number| (1..=MAX_NUMBERED_MINOR_PLANET).contains(&number))
        || is_canonical_minor_provisional(inner)
        || survey_parts(inner).is_ok()
}

/// Classify canonical unpacked ADES identity fields without rewriting the
/// submitted spelling.  The returned tuple is `(permID, provID, trkSub)`.
pub fn parse_ades_designation(
    designation: &str,
) -> Result<(Option<String>, Option<String>, Option<String>)> {
    if designation.is_empty() || designation.starts_with(' ') || designation.ends_with(' ') {
        return Err(invalid("ADES designation", designation));
    }

    if let Some(inner) = designation
        .strip_prefix('(')
        .and_then(|value| value.strip_suffix(')'))
    {
        if survey_parts(inner).is_ok() || is_canonical_minor_provisional(inner) {
            return Ok((None, Some(inner.to_string()), None));
        }
        return Err(MpcDesignationError::Value(
            "parenthesized MPC identity is not a provisional designation".to_string(),
        ));
    }

    if designation.bytes().all(|byte| byte.is_ascii_digit()) {
        let number = parse_canonical_u64(designation, "numbered minor-planet designation")?;
        if (1..=MAX_NUMBERED_MINOR_PLANET).contains(&number) {
            return Ok((Some(designation.to_string()), None, None));
        }
        return Err(MpcDesignationError::Value(format!(
            "numbered minor-planet designation out of range: {designation}"
        )));
    }

    if survey_parts(designation).is_ok() || is_canonical_minor_provisional(designation) {
        return Ok((None, Some(designation.to_string()), None));
    }

    // Legacy MPC display labels combine a permanent minor-planet number, an
    // optional proper name, and an optional parenthesized provisional designation.
    if let Some((number_text, display_tail)) = designation.split_once(' ') {
        if looks_like_malformed_minor_identity(number_text, display_tail) {
            return Err(MpcDesignationError::Value(
                "malformed minor-planet provisional designation".to_string(),
            ));
        }
        if let Ok(number) = parse_canonical_u64(number_text, "numbered minor-planet designation") {
            if (1..=MAX_NUMBERED_MINOR_PLANET).contains(&number)
                && !display_tail.is_empty()
                && !display_tail
                    .chars()
                    .any(|ch| matches!(ch, '|' | '\r' | '\n' | '\t'))
            {
                if display_tail.starts_with(' ')
                    || display_tail.ends_with(' ')
                    || display_tail.contains("  ")
                {
                    return Err(invalid("numbered minor-planet display label", designation));
                }
                let has_parenthesis = display_tail.chars().any(|ch| matches!(ch, '(' | ')'));
                let provisional = if has_parenthesis {
                    if !display_tail.ends_with(')') {
                        return Err(invalid("numbered minor-planet display label", designation));
                    }
                    let (name, candidate) = if let Some(candidate) = display_tail
                        .strip_prefix('(')
                        .and_then(|value| value.strip_suffix(')'))
                    {
                        (None, candidate)
                    } else {
                        let marker = display_tail.rfind(" (").ok_or_else(|| {
                            invalid("numbered minor-planet display label", designation)
                        })?;
                        let name = &display_tail[..marker];
                        let candidate = &display_tail[marker + 2..display_tail.len() - 1];
                        (Some(name), candidate)
                    };
                    if name.is_some_and(|value| !is_canonical_proper_name(value))
                        || candidate.chars().any(|ch| matches!(ch, '(' | ')'))
                    {
                        return Err(invalid("numbered minor-planet display label", designation));
                    }
                    if survey_parts(candidate).is_err()
                        && !is_canonical_minor_provisional(candidate)
                    {
                        return Err(MpcDesignationError::Value(
                            "parenthesized MPC identity is not a provisional designation"
                                .to_string(),
                        ));
                    }
                    Some(candidate.to_string())
                } else {
                    if !is_canonical_proper_name(display_tail) {
                        return Err(invalid("numbered minor-planet display label", designation));
                    }
                    None
                };
                return Ok((Some(number_text.to_string()), provisional, None));
            }
        }
    }

    if let Ok(parsed) = parse_comet_provisional(designation) {
        if let Some(number) = parsed.number {
            let permanent = format!("{number}{}", char::from(parsed.comet_type));
            let provisional = format!(
                "{}/{}",
                char::from(parsed.comet_type),
                designation.split_once('/').unwrap().1
            );
            return Ok((Some(permanent), Some(provisional), None));
        }
        return Ok((None, Some(designation.to_string()), None));
    }
    if let Ok((number, comet_type, fragment, _name)) = parse_numbered_comet(designation) {
        let fragment = fragment.map_or_else(String::new, |value| format!("-{value}"));
        return Ok((
            Some(format!("{number}{}{fragment}", char::from(comet_type))),
            None,
            None,
        ));
    }
    if is_ades_provisional_satellite(designation) {
        return Ok((None, Some(designation.to_string()), None));
    }
    if pack_permanent_satellite_designation(designation).is_ok() {
        return Ok((Some(designation.to_string()), None, None));
    }
    // ADES also represents satellites of numbered/provisional minor planets
    // directly, even though they have no planet-letter packed codec.
    if let Some(rest) = designation.strip_prefix('(') {
        if let Some((primary, satellite_number)) = rest.split_once(") ") {
            let valid_primary = parse_canonical_u64(primary, "satellite primary")
                .is_ok_and(|number| (1..=MAX_NUMBERED_MINOR_PLANET).contains(&number));
            if valid_primary && roman_value(satellite_number).is_some() {
                return Ok((Some(designation.to_string()), None, None));
            }
        }
    }

    // Reject official packed identities before the bounded tracking-ID fallback.
    // This is especially important for letter-prefixed and underscore-packed
    // minor planets, which otherwise look like plausible trkSub values.
    if unpack_mpc_designation(designation).is_ok() {
        return Err(MpcDesignationError::Value(format!(
            "packed MPC designations are not valid submitted ADES identities: {designation}"
        )));
    }
    if is_tracking_identifier(designation) {
        return Ok((None, None, Some(designation.to_string())));
    }
    Err(invalid("ADES designation", designation))
}

/// Legacy packed MPC epoch to ISOT string (TT scale).
/// See https://minorplanetcenter.net/iau/info/PackedDates.html.
pub fn unpack_mpc_date_isot(epoch_pf: &str) -> Result<String> {
    let chars: Vec<char> = epoch_pf.chars().collect();
    if chars.len() < 5 {
        return Err(MpcDesignationError::Index);
    }
    let year = python_int_base32(&chars[0..1].iter().collect::<String>())? * 100
        + python_int_compat(&chars[1..3].iter().collect::<String>())?;
    let month = python_int_base32(&chars[3..4].iter().collect::<String>())?;
    let day = python_int_base32(&chars[4..5].iter().collect::<String>())?;
    let mut isot = format!("{year}-{month:02}-{day:02}");

    if chars.len() > 5 {
        let tail: String = chars[5..].iter().collect();
        let fraction: f64 = format!("0.{tail}").parse().map_err(|_| {
            MpcDesignationError::Value(format!(
                "could not convert string to float: {}",
                py_repr(&format!(".{tail}"))
            ))
        })?;
        let hours = (24.0 * fraction) as i64;
        let minutes = (60.0 * (24.0 * fraction - hours as f64)) as i64;
        let seconds = 3600.0 * (24.0 * fraction - hours as f64 - minutes as f64 / 60.0);
        isot.push_str(&format!("T{hours:02}:{minutes:02}:{seconds:09.6}"));
    }
    Ok(isot)
}

fn py_repr(value: &str) -> String {
    format!("'{}'", value.replace('\\', "\\\\").replace('\'', "\\'"))
}

fn python_int_compat(value: &str) -> Result<i64> {
    let trimmed = value.trim();
    let ok = !trimmed.is_empty()
        && trimmed
            .strip_prefix(['+', '-'])
            .unwrap_or(trimmed)
            .chars()
            .all(|character| character.is_ascii_digit())
        && trimmed
            .strip_prefix(['+', '-'])
            .is_none_or(|rest| !rest.is_empty());
    if !ok {
        return Err(MpcDesignationError::Value(format!(
            "invalid literal for int() with base 10: {}",
            py_repr(value)
        )));
    }
    trimmed.parse::<i64>().map_err(|_| {
        MpcDesignationError::Value(format!(
            "invalid literal for int() with base 10: {}",
            py_repr(value)
        ))
    })
}

fn python_int_base32(value: &str) -> Result<i64> {
    let trimmed = value.trim();
    let ok = !trimmed.is_empty()
        && trimmed
            .chars()
            .all(|c| c.is_ascii_digit() || matches!(c.to_ascii_lowercase(), 'a'..='v'));
    if !ok {
        return Err(MpcDesignationError::Value(format!(
            "invalid literal for int() with base 32: {}",
            py_repr(value)
        )));
    }
    i64::from_str_radix(&trimmed.to_ascii_lowercase(), 32).map_err(|_| {
        MpcDesignationError::Value(format!(
            "invalid literal for int() with base 32: {}",
            py_repr(value)
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_date_compatibility_is_retained() {
        assert_eq!(unpack_mpc_date_isot("J9611").unwrap(), "1996-01-01");
        assert_eq!(
            unpack_mpc_date_isot("J969U75").unwrap(),
            "1996-09-30T18:00:00.000000"
        );
    }

    #[test]
    fn minor_planet_boundaries_and_current_extended_examples() {
        let cases = [
            ("1", "00001"),
            ("99999", "99999"),
            ("100000", "A0000"),
            ("619999", "z9999"),
            ("620000", "~0000"),
            ("15396335", "~zzzz"),
            ("1995 XA", "J95X00A"),
            ("2025 OY619", "K25Oz9Y"),
            ("2015 BX634", "_FB0060"),
            ("2015 BZ631", "_FB004p"),
            ("2024 AB631", "_OA004S"),
            ("2025 OY625", "_PO002O"),
            ("2025 OT677", "_PO00NH"),
            ("2040 P-L", "PLS2040"),
            ("A908 CJ", "J08C00J"),
        ];
        for (unpacked, packed) in cases {
            assert_eq!(pack_mpc_designation(unpacked).unwrap(), packed);
        }
        assert_eq!(
            unpack_provisional_designation("J08C00J").unwrap(),
            "A908 CJ"
        );
        assert_eq!(unpack_mpc_designation("_FB004p").unwrap(), "2015 BZ631");
        assert_eq!(unpack_mpc_designation("PLS2040").unwrap(), "2040 P-L");
        assert_eq!(unpack_mpc_designation("~zzzz").unwrap(), "15396335");
    }

    #[test]
    fn comet_and_satellite_round_trips() {
        let cases = [
            ("1P", "0001P"),
            ("73P-AA", "0073Paa"),
            ("C/1995 O1", "CJ95O010"),
            ("D/1993 F2-B", "DJ93F02b"),
            ("A/2017 U1", "AK17U010"),
            ("1P/1986 F1", "0001PJ86F010"),
            ("C/240 V1", "C240V010"),
            ("C/-43 K1", "C/56K010"),
            ("S/2019 S 22", "SK19S220"),
            ("Jupiter XIII", "J013S"),
        ];
        for (unpacked, packed) in cases {
            assert_eq!(pack_mpc_designation(unpacked).unwrap(), packed);
            assert_eq!(unpack_mpc_designation(packed).unwrap(), unpacked);
        }
    }

    #[test]
    fn rejects_noncanonical_and_out_of_range_values() {
        for value in ["0", "0001", "1893 AP", "2015 BI620", "2062 AA620", "0 P-L"] {
            assert!(pack_mpc_designation(value).is_err(), "{value}");
        }
        for value in ["00000", "~zzzzx", "_!A0000", "0000P", "J000S", "SK19S000"] {
            assert!(unpack_mpc_designation(value).is_err(), "{value}");
        }
    }

    #[test]
    fn ades_minor_provisionals_require_packable_boundaries_in_every_context() {
        for designation in ["1999 AA620", "2062 AA620", "2025 AA591674"] {
            for value in [
                designation.to_string(),
                format!("({designation})"),
                format!("17032 Edlu ({designation})"),
                format!("S/2000 ({designation}) 1"),
            ] {
                assert!(parse_ades_designation(&value).is_err(), "{value}");
            }
        }

        for designation in ["2061 AZ620", "2029 FL591673"] {
            assert_eq!(
                parse_ades_designation(designation).unwrap(),
                (None, Some(designation.to_string()), None)
            );
            assert_eq!(
                parse_ades_designation(&format!("({designation})")).unwrap(),
                (None, Some(designation.to_string()), None)
            );
            assert_eq!(
                parse_ades_designation(&format!("17032 Edlu ({designation})")).unwrap(),
                (
                    Some("17032".to_string()),
                    Some(designation.to_string()),
                    None
                )
            );
            let satellite = format!("S/2000 ({designation}) 1");
            assert_eq!(
                parse_ades_designation(&satellite).unwrap(),
                (None, Some(satellite), None)
            );
        }
    }

    #[test]
    fn ades_planetary_satellite_numbers_must_be_packable() {
        assert_eq!(
            pack_provisional_satellite_designation("S/2019 S 619").unwrap(),
            "SK19Sz90"
        );
        assert_eq!(
            parse_ades_designation("S/2019 S 619").unwrap(),
            (None, Some("S/2019 S 619".to_string()), None)
        );
        for designation in ["S/2019 S 620", "S/2019 S 999999", "S/2019 S 0619"] {
            assert!(
                parse_ades_designation(designation).is_err(),
                "{designation}"
            );
        }

        // Parenthesized minor-planet primaries are an ADES form without a
        // planet-letter packed codec, so their positive canonical ordinal is
        // intentionally not constrained by the two-character packed field.
        assert_eq!(
            parse_ades_designation("S/2019 (134340) 620").unwrap(),
            (None, Some("S/2019 (134340) 620".to_string()), None)
        );
    }

    #[test]
    fn ades_classification_rejects_malformed_identity_like_labels_without_panicking() {
        for designation in [
            "2015 Bx",
            "1995 X A",
            "2040 P-l",
            "2015 BXA",
            "2015 Bxa",
            "2015 BXAB",
            "1908 Cj",
            "Jupiter IIIIIIIIIIIX",
            "1P/1986",
        ] {
            assert!(
                parse_ades_designation(designation).is_err(),
                "{designation}"
            );
        }
        assert!(roman_value("IIIIIIIIIIX").is_none());

        for designation in [
            "2015 BZ631",
            "2040 P-L",
            "17032 Edlu (1999 FM9)",
            "1036 Ganymed",
            "1700 AAS",
            "5000 IAU",
            "3654 AAS",
            "2062 Aten",
            "1P/Halley",
            "1P/1986 F1",
        ] {
            assert!(parse_ades_designation(designation).is_ok(), "{designation}");
        }
    }

    #[test]
    fn ades_classification_is_strict() {
        assert_eq!(
            parse_ades_designation("2015 BZ631").unwrap(),
            (None, Some("2015 BZ631".to_string()), None)
        );
        assert_eq!(
            parse_ades_designation("1P/1986 F1").unwrap(),
            (Some("1P".to_string()), Some("P/1986 F1".to_string()), None)
        );
        assert_eq!(
            parse_ades_designation("MIRA25").unwrap(),
            (None, None, Some("MIRA25".to_string()))
        );
        for packed in ["A0345", "_FB004p", "K24A00A", "PLS2040", "0001P", "J013S"] {
            assert!(parse_ades_designation(packed).is_err(), "{packed}");
        }
    }
}
