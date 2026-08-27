//! MPC packed-designation conversions (bead personal-cmy 'W11 helper gaps').
//!
//! Ports `adam_core.utils.mpc` pack/unpack designation helpers to Rust with
//! legacy-exact semantics, including the quirks the Python dispatch chain
//! depends on:
//!
//! * only `ValueError`s are swallowed when trying successive forms, so bad
//!   base-62 characters surface as `KeyError` and out-of-range indexing as
//!   `IndexError`, exactly like the legacy code;
//! * `int()` parse failures reproduce CPython's
//!   `invalid literal for int() with base 10: '...'` message;
//! * legacy validation-order quirks are preserved (e.g. the provisional
//!   packer checks only the first three year characters, the survey unpacker
//!   parses the number before validating the survey prefix, and one legacy
//!   `raise ValueError()` carries an empty message).

const BASE62: &[u8; 62] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

/// Error kinds mirroring the Python exception types the legacy code raises.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MpcDesignationError {
    /// Python `ValueError` with this exact message (may be empty).
    Value(String),
    /// Python `KeyError` for this key (str() renders it quoted).
    Key(String),
    /// Python `IndexError: string index out of range`.
    Index,
}

impl std::fmt::Display for MpcDesignationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Value(message) => write!(f, "ValueError: {message}"),
            Self::Key(key) => write!(f, "KeyError: '{key}'"),
            Self::Index => write!(f, "IndexError: string index out of range"),
        }
    }
}

type MpcResult<T> = Result<T, MpcDesignationError>;

fn py_repr(value: &str) -> String {
    format!("'{}'", value.replace('\\', "\\\\").replace('\'', "\\'"))
}

/// Python `int(s)` for base 10 (ASCII digits, optional sign, surrounding
/// whitespace), with the CPython error message.
fn python_int(value: &str) -> MpcResult<i64> {
    let trimmed = value.trim();
    let ok = !trimmed.is_empty()
        && trimmed
            .strip_prefix(['+', '-'])
            .unwrap_or(trimmed)
            .chars()
            .all(|c| c.is_ascii_digit())
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

/// Python `str.isdecimal()` restricted to ASCII digits (documented deviation:
/// non-ASCII Unicode decimals are not supported).
fn is_decimal(value: &str) -> bool {
    !value.is_empty() && value.chars().all(|c| c.is_ascii_digit())
}

fn base62_char(index: i64) -> MpcResult<char> {
    if !(0..62).contains(&index) {
        return Err(MpcDesignationError::Index);
    }
    Ok(BASE62[index as usize] as char)
}

fn base62_value(c: char) -> MpcResult<i64> {
    BASE62
        .iter()
        .position(|&b| b as char == c)
        .map(|index| index as i64)
        .ok_or_else(|| MpcDesignationError::Key(c.to_string()))
}

fn zfill(value: &str, width: usize) -> String {
    if value.chars().count() >= width {
        return value.to_string();
    }
    let (sign, digits) = match value.strip_prefix(['+', '-']) {
        Some(rest) => (&value[..1], rest),
        None => ("", value),
    };
    let pad = width - value.chars().count();
    format!("{sign}{}{digits}", "0".repeat(pad))
}

pub fn pack_numbered_designation(designation: &str) -> MpcResult<String> {
    let number = python_int(designation)?;
    if number > 15_396_335 {
        return Err(MpcDesignationError::Value(
            "Numbered designation is too large. Maximum supported is 15396335.".to_string(),
        ));
    }

    if number <= 99_999 {
        Ok(format!("{number:05}"))
    } else if (100_000..=619_999).contains(&number) {
        let bigpart = number / 10_000;
        let remainder = number % 10_000;
        Ok(format!("{}{remainder:04}", base62_char(bigpart)?))
    } else {
        let mut x = number - 620_000;
        let mut digits: Vec<char> = Vec::new();
        while x != 0 {
            digits.push(base62_char(x % 62)?);
            x /= 62;
        }
        digits.reverse();
        let packed: String = digits.into_iter().collect();
        Ok(format!("~{}", zfill(&packed, 4)))
    }
}

pub fn pack_provisional_designation(designation: &str) -> MpcResult<String> {
    let chars: Vec<char> = designation.chars().collect();
    if chars.len() < 6 {
        return Err(MpcDesignationError::Value(
            "Provisional designations should be at least 6 characters long.".to_string(),
        ));
    }
    // Legacy quirk: only the first three characters are checked, despite the
    // error message talking about four.
    let first3: String = chars[..3].iter().collect();
    if !is_decimal(&first3) {
        return Err(MpcDesignationError::Value(
            "Expected the first 4 characters of the provisional designation to be a year."
                .to_string(),
        ));
    }
    if chars[4] != ' ' {
        return Err(MpcDesignationError::Value(
            "Expected the 5th character of the provisional designation to be a space.".to_string(),
        ));
    }
    if designation.contains('-') {
        return Err(MpcDesignationError::Value(
            "Provisional designations cannot contain a hyphen.".to_string(),
        ));
    }

    let century: String = chars[0..2].iter().collect();
    let year_tail: String = chars[2..4].iter().collect();
    let year = format!("{}{year_tail}", base62_char(python_int(&century)?)?);
    let letter1 = chars[5];
    let letter2 = *chars.get(6).ok_or(MpcDesignationError::Index)?;
    let cycle: String = chars[7..].iter().collect();

    if letter1 == 'I' || letter1 == 'Z' {
        return Err(MpcDesignationError::Value(
            "Half-month letters cannot be I or Z.".to_string(),
        ));
    }
    if letter1.is_ascii_digit() || letter2.is_ascii_digit() {
        return Err(MpcDesignationError::Value(
            "Invalid provisional designation.".to_string(),
        ));
    }

    let mut cycle_pf = "00".to_string();
    if !cycle.is_empty() {
        let cycle_int = python_int(&cycle)?;
        if cycle_int <= 99 {
            cycle_pf = zfill(&cycle_int.to_string(), 2);
        } else {
            cycle_pf = format!("{}{}", base62_char(cycle_int / 10)?, cycle_int % 10);
        }
    }

    Ok(format!("{year}{letter1}{cycle_pf}{letter2}"))
}

pub fn pack_survey_designation(designation: &str) -> MpcResult<String> {
    let chars: Vec<char> = designation.chars().collect();
    let number: String = chars.iter().take(4).collect();
    let survey: String = chars.iter().skip(5).collect();

    let survey_pf = if survey == "P-L" {
        "PLS".to_string()
    } else if survey.starts_with("T-")
        && matches!(survey.chars().nth(2), Some('1') | Some('2') | Some('3'))
    {
        format!("T{}S", survey.chars().nth(2).unwrap())
    } else {
        return Err(MpcDesignationError::Value(
            "Survey designations must start with P-L, T-1, T-2, T-3.".to_string(),
        ));
    };

    Ok(format!("{survey_pf}{}", zfill(&number, 4)))
}

pub fn pack_mpc_designation(designation: &str) -> MpcResult<String> {
    // The legacy chain swallows only ValueError; KeyError/IndexError escape.
    for attempt in [
        pack_numbered_designation(designation),
        pack_provisional_designation(designation),
        pack_survey_designation(designation),
    ] {
        match attempt {
            Ok(packed) => return Ok(packed),
            Err(MpcDesignationError::Value(_)) => continue,
            Err(other) => return Err(other),
        }
    }
    Err(MpcDesignationError::Value(format!(
        "Unpacked designation '{designation}' could not be packed.\n\
         It could not be recognized as any of the following:\n\
         \x20- a numbered object (e.g. '3202', '203289', '3140113')\n\
         \x20- a provisional designation (e.g. '1998 SV127', '2008 AA360')\n\
         \x20- a survey designation (e.g. '2040 P-L', '3138 T-1')"
    )))
}

pub fn unpack_numbered_designation(designation_pf: &str) -> MpcResult<String> {
    let chars: Vec<char> = designation_pf.chars().collect();
    let number: i64;

    if is_decimal(designation_pf) {
        number = python_int(designation_pf)?;
    } else if *chars.first().ok_or(MpcDesignationError::Index)? == '~' {
        let mut value = 620_000_i64;
        let digits = &chars[1..];
        for (i, &c) in digits.iter().enumerate() {
            let power = (digits.len() - (i + 1)) as u32;
            value += base62_value(c)? * 62_i64.pow(power);
        }
        number = value;
    } else {
        let rest: String = chars[1..].iter().collect();
        number = base62_value(chars[0])? * 10_000 + python_int(&rest)?;
    }

    Ok(number.to_string())
}

pub fn unpack_provisional_designation(designation_pf: &str) -> MpcResult<String> {
    let chars: Vec<char> = designation_pf.chars().collect();
    if chars.len() != 7 {
        return Err(MpcDesignationError::Value(
            "Provisional designation must be 7 characters long.".to_string(),
        ));
    }
    if !chars[1].is_ascii_digit() || !chars[2].is_ascii_digit() {
        return Err(MpcDesignationError::Value(
            "Provisional designation must have a year.".to_string(),
        ));
    }
    let year_tail: String = chars[1..3].iter().collect();
    let year = (base62_value(chars[0])? * 100 + python_int(&year_tail)?).to_string();
    let letter1 = chars[3];
    let letter2 = chars[6];
    if letter1.is_ascii_digit() || letter2.is_ascii_digit() {
        // Legacy raises a bare `ValueError()` here (empty message).
        return Err(MpcDesignationError::Value(String::new()));
    }
    let cycle1 = chars[4];
    let cycle2 = chars[5];

    let number = base62_value(cycle1)? * 10 + base62_value(cycle2)?;
    let number_str = if number == 0 {
        String::new()
    } else {
        number.to_string()
    };

    Ok(format!("{year} {letter1}{letter2}{number_str}"))
}

pub fn unpack_survey_designation(designation_pf: &str) -> MpcResult<String> {
    let chars: Vec<char> = designation_pf.chars().collect();
    // Legacy parses the number before validating the survey prefix.
    let number_str: String = chars.iter().skip(3).take(5).collect();
    let number = python_int(&number_str)?;
    let survey_pf: String = chars.iter().take(3).collect();
    if !matches!(survey_pf.as_str(), "PLS" | "T1S" | "T2S" | "T3S") {
        return Err(MpcDesignationError::Value(
            "Packed survey designation must start with PLS, T1S, T2S, or T3S.".to_string(),
        ));
    }

    let survey = if survey_pf == "PLS" {
        "P-L".to_string()
    } else {
        format!("T-{}", survey_pf.chars().nth(1).unwrap())
    };

    Ok(format!("{number} {survey}"))
}

pub fn unpack_mpc_designation(designation_pf: &str) -> MpcResult<String> {
    for attempt in [
        unpack_numbered_designation(designation_pf),
        unpack_provisional_designation(designation_pf),
        unpack_survey_designation(designation_pf),
    ] {
        match attempt {
            Ok(unpacked) => return Ok(unpacked),
            Err(MpcDesignationError::Value(_)) => continue,
            Err(other) => return Err(other),
        }
    }
    Err(MpcDesignationError::Value(format!(
        "Packed form designation '{designation_pf}' could not be unpacked.\n\
         It could not be recognized as any of the following:\n\
         \x20- a numbered object (e.g. '03202', 'K3289', '~AZaz')\n\
         \x20- a provisional designation (e.g. 'J98SC7V', 'K08Aa0A')\n\
         \x20- a survey designation (e.g. 'PLS2040', 'T1S3138')"
    )))
}

/// Legacy `_unpack_mpc_date` (packed MPC epoch -> ISOT string, TT scale).
/// See https://minorplanetcenter.net/iau/info/PackedDates.html.
pub fn unpack_mpc_date_isot(epoch_pf: &str) -> MpcResult<String> {
    let chars: Vec<char> = epoch_pf.chars().collect();
    if chars.len() < 5 {
        return Err(MpcDesignationError::Index);
    }
    let year = python_int_base32(&chars[0..1].iter().collect::<String>())? * 100
        + python_int(&chars[1..3].iter().collect::<String>())?;
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

/// Python `int(s, base=32)` for single characters (digits + a-v, case
/// insensitive), with the CPython error message.
fn python_int_base32(value: &str) -> MpcResult<i64> {
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

    /// The MPC documentation example table (same panel the legacy tests use).
    const NUMBERED: [(&str, &str); 9] = [
        ("3202", "03202"),
        ("50000", "50000"),
        ("100345", "A0345"),
        ("360017", "a0017"),
        ("203289", "K3289"),
        ("620000", "~0000"),
        ("620061", "~000z"),
        ("3140113", "~AZaz"),
        ("15396335", "~zzzz"),
    ];
    const PROVISIONAL: [(&str, &str); 9] = [
        ("1995 XA", "J95X00A"),
        ("1995 XL1", "J95X01L"),
        ("1995 FB13", "J95F13B"),
        ("1998 SQ108", "J98SA8Q"),
        ("1998 SV127", "J98SC7V"),
        ("1998 SS162", "J98SG2S"),
        ("2099 AZ193", "K99AJ3Z"),
        ("2008 AA360", "K08Aa0A"),
        ("2007 TA418", "K07Tf8A"),
    ];
    const SURVEY: [(&str, &str); 4] = [
        ("2040 P-L", "PLS2040"),
        ("3138 T-1", "T1S3138"),
        ("1010 T-2", "T2S1010"),
        ("4101 T-3", "T3S4101"),
    ];

    #[test]
    fn mpc_example_table_round_trips() {
        for (unpacked, packed) in NUMBERED.iter().chain(&PROVISIONAL).chain(&SURVEY) {
            assert_eq!(pack_mpc_designation(unpacked).unwrap(), *packed);
            assert_eq!(unpack_mpc_designation(packed).unwrap(), *unpacked);
        }
    }

    #[test]
    fn numbered_boundaries() {
        assert_eq!(pack_numbered_designation("99999").unwrap(), "99999");
        assert_eq!(pack_numbered_designation("100000").unwrap(), "A0000");
        assert_eq!(pack_numbered_designation("619999").unwrap(), "z9999");
        assert_eq!(pack_numbered_designation("620000").unwrap(), "~0000");
        assert!(matches!(
            pack_numbered_designation("15396336"),
            Err(MpcDesignationError::Value(_))
        ));
        for number in [0_i64, 1, 99_999, 100_000, 619_999, 620_000, 15_396_335] {
            let packed = pack_numbered_designation(&number.to_string()).unwrap();
            assert_eq!(
                unpack_numbered_designation(&packed).unwrap(),
                number.to_string()
            );
        }
    }

    #[test]
    fn error_kinds_match_legacy_semantics() {
        // Bad base-62 characters surface as KeyError, escaping the dispatch.
        assert_eq!(
            unpack_mpc_designation("!0345"),
            Err(MpcDesignationError::Key("!".to_string()))
        );
        // Empty input hits string indexing -> IndexError.
        assert_eq!(
            unpack_numbered_designation(""),
            Err(MpcDesignationError::Index)
        );
        // Six-character provisional input hits designation[6] -> IndexError.
        assert_eq!(
            pack_provisional_designation("1995 X"),
            Err(MpcDesignationError::Index)
        );
        // Cycle >= 620 indexes past the base-62 table -> IndexError.
        assert_eq!(
            pack_provisional_designation("2008 AA6200"),
            Err(MpcDesignationError::Index)
        );
        // Bare ValueError() with an empty message.
        assert_eq!(
            unpack_provisional_designation("J95X001"),
            Err(MpcDesignationError::Value(String::new()))
        );
        // int() failure message matches CPython.
        assert_eq!(
            pack_numbered_designation("1995 XA"),
            Err(MpcDesignationError::Value(
                "invalid literal for int() with base 10: '1995 XA'".to_string()
            ))
        );
    }

    #[test]
    fn negative_number_quirk_matches_python_format() {
        // Legacy int + "{:05}" happily packs negative numbers.
        assert_eq!(pack_numbered_designation("-5").unwrap(), "-0005");
    }
}
