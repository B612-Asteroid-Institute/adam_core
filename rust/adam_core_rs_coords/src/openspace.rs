//! OpenSpace Lua/asset text rendering (bead personal-cmy.28).
//!
//! Python remains responsible for building the OpenSpace dataclass object graph
//! and orbit-derived values; this module owns the byte-compatible text renderer
//! behind the canonical public Python names. It mirrors the legacy `LuaDict`
//! formatting exactly, including PascalCase field names, the special `GUI`
//! spelling, bool/string/tuple/list formatting, nested indentation, and
//! `asset.resource("...")` resource expressions.

use crate::types::{SchemaError, SchemaResult};
use serde_json::Value;
use std::fmt::Write as _;

fn invalid(message: String) -> SchemaError {
    SchemaError::InvalidRecordBatch(message)
}

fn pascal_case(value: &str) -> String {
    value
        .split('_')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                Some(first) => {
                    let mut out = first.to_uppercase().collect::<String>();
                    out.push_str(&chars.as_str().to_lowercase());
                    out
                }
                None => String::new(),
            }
        })
        .collect::<String>()
}

fn required_str<'a>(value: &'a Value, key: &str) -> SchemaResult<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| invalid(format!("OpenSpace payload missing string key {key:?}")))
}

fn required_array<'a>(value: &'a Value, key: &str) -> SchemaResult<&'a Vec<Value>> {
    value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| invalid(format!("OpenSpace payload missing array key {key:?}")))
}

fn render_node(value: &Value, indent: usize) -> SchemaResult<String> {
    let kind = required_str(value, "kind")?;
    match kind {
        "dict" => render_dict(value, indent),
        "resource" => Ok(format!(
            "asset.resource(\"{}\")",
            required_str(value, "path")?
        )),
        "string" => Ok(format!("\"{}\"", required_str(value, "value")?)),
        "raw" => Ok(required_str(value, "value")?.to_string()),
        "bool" => Ok(
            if value.get("value").and_then(Value::as_bool).unwrap_or(false) {
                "true".to_string()
            } else {
                "false".to_string()
            },
        ),
        "tuple" => {
            let values = required_array(value, "values")?
                .iter()
                .map(|v| {
                    v.as_str().ok_or_else(|| {
                        invalid("OpenSpace tuple values must be strings".to_string())
                    })
                })
                .collect::<SchemaResult<Vec<_>>>()?;
            Ok(format!("{{{}}}", values.join(", ")))
        }
        "list" => {
            let values = required_array(value, "values")?
                .iter()
                .map(|v| {
                    v.as_str()
                        .ok_or_else(|| invalid("OpenSpace list values must be strings".to_string()))
                })
                .collect::<SchemaResult<Vec<_>>>()?;
            // Legacy LuaDict rendered a Python list as a one-element Lua table
            // containing the comma-joined items as a string.
            Ok(format!("{{\"{}\"}}", values.join(", ")))
        }
        other => Err(invalid(format!(
            "unknown OpenSpace Lua payload kind {other:?}"
        ))),
    }
}

fn render_dict(value: &Value, indent: usize) -> SchemaResult<String> {
    let mut lua = String::from("{\n");
    for field in required_array(value, "fields")? {
        let key = required_str(field, "key")?;
        let node = field
            .get("value")
            .ok_or_else(|| invalid(format!("OpenSpace field {key:?} missing value")))?;
        let rendered = render_node(node, indent + 4)?;
        let key_str = if key == "gui" {
            "GUI".to_string()
        } else {
            pascal_case(key)
        };
        let _ = writeln!(lua, "{}{} = {},", " ".repeat(indent), key_str, rendered);
    }
    lua.push_str(&" ".repeat(indent.saturating_sub(4)));
    lua.push('}');
    Ok(lua)
}

pub fn openspace_lua_to_string(payload_json: &str, indent: usize) -> SchemaResult<String> {
    let payload: Value = serde_json::from_str(payload_json)
        .map_err(|err| invalid(format!("invalid OpenSpace Lua payload: {err}")))?;
    render_node(&payload, indent)
}

pub fn openspace_create_initialization(assets: &[String]) -> String {
    let mut initialization = vec!["asset.onInitialize(function ()".to_string()];
    let mut deinitialization = vec!["asset.onDeinitialize(function ()".to_string()];
    for asset in assets {
        initialization.push(format!("  openspace.addSceneGraphNode({asset});"));
        deinitialization.push(format!("  openspace.removeSceneGraphNode({asset});"));
    }
    initialization.push("end)".to_string());
    deinitialization.push("end)".to_string());
    format!(
        "{}\n{}",
        initialization.join("\n"),
        deinitialization.join("\n")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_legacy_lua_dict_payload() {
        let payload = r#"{
            "kind": "dict",
            "fields": [
                {"key": "identifier", "value": {"kind": "string", "value": "Obj"}},
                {"key": "gui", "value": {"kind": "dict", "fields": [
                    {"key": "name", "value": {"kind": "string", "value": "Name"}},
                    {"key": "enabled", "value": {"kind": "bool", "value": true}}
                ]}},
                {"key": "color", "value": {"kind": "tuple", "values": ["1.0", "0.5", "0.25"]}},
                {"key": "tag", "value": {"kind": "list", "values": ["alpha", "beta"]}},
                {"key": "path", "value": {"kind": "resource", "path": "x.csv"}}
            ]
        }"#;
        let expected = "{\n    Identifier = \"Obj\",\n    GUI = {\n        Name = \"Name\",\n        Enabled = true,\n    },\n    Color = {1.0, 0.5, 0.25},\n    Tag = {\"alpha, beta\"},\n    Path = asset.resource(\"x.csv\"),\n}";
        assert_eq!(openspace_lua_to_string(payload, 4).unwrap(), expected);
    }

    #[test]
    fn initialization_layout_matches_legacy() {
        assert_eq!(
            openspace_create_initialization(&["Object".to_string(), "Head".to_string()]),
            "asset.onInitialize(function ()\n  openspace.addSceneGraphNode(Object);\n  openspace.addSceneGraphNode(Head);\nend)\nasset.onDeinitialize(function ()\n  openspace.removeSceneGraphNode(Object);\n  openspace.removeSceneGraphNode(Head);\nend)"
        );
    }
}
