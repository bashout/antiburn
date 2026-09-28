//! Mistral Vibe unified session store v1 bundle reader.
//!
//! A Mistral Vibe session is a directory that holds a `meta.json` metadata
//! file, a `CURRENT` generation pointer, a `journal/` event log, and
//! `generations/` snapshots. `CURRENT` pins the store format and names the
//! newest generation. The journal holds hash-chained rows ordered by
//! `sequence`; the newest projection state in those rows carries the
//! cumulative session token total. The newest generation snapshot carries
//! the model alias and the reasoning effort in `runtime-state.json`.
//!
//! This reader retains only model, token, tool, thinking, and order facts.
//! It does not retain prompts, message content, tool input, tool results,
//! paths, or permissions. The token total is a session cumulative, so it is
//! emitted once as a single session-level usage event. The model alias is
//! written only when a session pins one, so model facts are conditional.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use serde_json::Value;

use crate::analysis::framing::{BoundedJsonlReader, FramedRecord, PartialReason};
use crate::analysis::interface::{
    RawSource, RecordSink, SessionCollector, SessionInput, SessionReader, SessionSummary,
    VisitOutcome,
};
use crate::analysis::model::{
    NormalizedEvent, NormalizedSession, Role, ToolCall, ToolCategory, Usage,
};
use crate::analysis::{SourceCapabilities, SourceFormat};

/// The store format string this reader pins. `CURRENT` carries it beside a
/// minor version; see [`PINNED_STORE_FORMAT_MINOR`].
const PINNED_STORE_FORMAT: &str = "mistral.vibe.unified-session-store/v1";

/// The store format minor this reader is characterized against. A store
/// written by any other minor fails closed until it is characterized.
const PINNED_STORE_FORMAT_MINOR: i64 = 7;

/// The journal row types this reader recognizes. Any other row type marks
/// the source partial, because an unrecognized row can change the facts.
const KNOWN_ROW_TYPES: &[&str] = &[
    "projection_delta",
    "core_input",
    "action_intent",
    "action_result",
    "callback_registered",
    "callback_resolved",
    "command_reserved",
    "receipt_succeeded",
];

pub struct MistralVibeSessionReader;

impl SessionReader for MistralVibeSessionReader {
    fn agent(&self) -> &'static str {
        "mistral-vibe"
    }

    fn capabilities(&self, _input: &SessionInput) -> SourceCapabilities {
        SourceCapabilities::mistral_vibe()
    }

    fn normalize(&self, input: &SessionInput) -> anyhow::Result<NormalizedSession> {
        let mut collector = SessionCollector::new(input.agent.clone(), input.session_id.clone());
        self.visit(input, &mut collector)?;
        collector.into_session()
    }

    fn visit(
        &self,
        input: &SessionInput,
        sink: &mut dyn RecordSink,
    ) -> anyhow::Result<VisitOutcome> {
        if input.source_format != SourceFormat::MistralVibeUnifiedStoreV1
            && input.source_format != SourceFormat::Uncharacterized
        {
            sink.record(crate::analysis::NormalizedRecord::Unusable(
                PartialReason::MalformedRecord,
            ));
            sink.finish(SessionSummary::default());
            return Ok(VisitOutcome::Unvalidated);
        }
        let RawSource::MistralVibeUnifiedBundle { session_dir } = &input.source else {
            anyhow::bail!("Mistral Vibe requires a unified session store directory");
        };
        let metadata: Value = serde_json::from_reader(File::open(session_dir.join("meta.json"))?)?;
        let (session_id, metadata_model, metadata_thinking) = validate_metadata(&metadata)?;
        let current: Value = serde_json::from_reader(File::open(session_dir.join("CURRENT"))?)?;
        let generation = validate_current(&current, &session_id)?;

        // The newest generation snapshot carries the model alias and the
        // reasoning effort. A store that has not written a snapshot yet
        // keeps both facts unknown rather than failing the session.
        let (model, thinking_level) = match read_snapshot_facts(session_dir, &generation) {
            Some((snapshot_model, snapshot_thinking)) => (
                snapshot_model.or(metadata_model),
                snapshot_thinking.or(metadata_thinking),
            ),
            None => (metadata_model, metadata_thinking),
        };

        let mut invalid = false;
        let mut cumulative = CumulativeUsage::default();
        for path in journal_files(session_dir)? {
            let mut framed = BoundedJsonlReader::new(BufReader::new(File::open(path)?));
            while let Some(record) = framed.next_record(&|| false) {
                match record {
                    FramedRecord::Skipped(skip) => {
                        invalid = true;
                        sink.record(crate::analysis::NormalizedRecord::Unusable(
                            skip.partial_reason(),
                        ));
                    }
                    FramedRecord::Complete { bytes, .. } => {
                        match serde_json::from_slice::<Value>(bytes) {
                            Ok(value) => {
                                if !observe_journal_row(
                                    &value,
                                    &model,
                                    &thinking_level,
                                    &mut cumulative,
                                    sink,
                                ) {
                                    invalid = true;
                                    sink.record(crate::analysis::NormalizedRecord::Unusable(
                                        PartialReason::MalformedRecord,
                                    ));
                                }
                            }
                            Err(_) => {
                                invalid = true;
                                sink.record(crate::analysis::NormalizedRecord::Unusable(
                                    PartialReason::MalformedRecord,
                                ));
                            }
                        }
                    }
                }
            }
        }
        if let Some(event) = cumulative.into_event(&model, &thinking_level) {
            sink.record(crate::analysis::NormalizedRecord::MetricsEvent(Box::new(
                event,
            )));
        }
        let mut summary = SessionSummary {
            model,
            ..SessionSummary::default()
        };
        if invalid {
            summary.coverage_gaps.push(PartialReason::MalformedRecord);
        }
        sink.finish(summary);
        Ok(VisitOutcome::Unvalidated)
    }
}

/// Reads the session identity from `meta.json`. Returns the session id with
/// any model alias and thinking level the metadata carries directly.
fn validate_metadata(metadata: &Value) -> anyhow::Result<(String, Option<String>, Option<String>)> {
    let object = metadata
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("Mistral Vibe metadata is not an object"))?;
    let Some(session_id) = object.get("session_id").and_then(Value::as_str) else {
        anyhow::bail!("Mistral Vibe metadata does not match the session contract");
    };
    if !metadata
        .pointer("/environment/working_directory")
        .is_some_and(Value::is_string)
        || object.get("start_time").and_then(Value::as_str).is_none()
    {
        anyhow::bail!("Mistral Vibe metadata does not match the session contract");
    }
    let model = metadata
        .pointer("/config/active_model")
        .and_then(Value::as_str)
        .filter(|model| !model.is_empty())
        .map(catalog_model_id);
    let thinking_level = metadata
        .pointer("/config/thinking")
        .and_then(Value::as_str)
        .filter(|level| !level.is_empty())
        .map(str::to_owned);
    Ok((session_id.to_owned(), model, thinking_level))
}

/// Reads `CURRENT` and returns the generation the store points at. The
/// store format and the session id must match the characterized contract.
fn validate_current(current: &Value, session_id: &str) -> anyhow::Result<String> {
    let object = current
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("Mistral Vibe CURRENT is not an object"))?;
    if object.get("store_format").and_then(Value::as_str) != Some(PINNED_STORE_FORMAT)
        || object.get("store_format_minor").and_then(Value::as_i64)
            != Some(PINNED_STORE_FORMAT_MINOR)
        || object.get("session_id").and_then(Value::as_str) != Some(session_id)
    {
        anyhow::bail!("Mistral Vibe CURRENT does not match the store contract");
    }
    let Some(generation) = object.get("generation").and_then(Value::as_str) else {
        anyhow::bail!("Mistral Vibe CURRENT does not name a generation");
    };
    if generation.is_empty() || !generation.bytes().all(|b| b.is_ascii_digit()) {
        anyhow::bail!("Mistral Vibe CURRENT names an invalid generation");
    }
    Ok(generation.to_owned())
}

/// The model alias and reasoning effort recorded by the newest generation
/// snapshot. Returns `None` when the snapshot has not been written yet.
fn read_snapshot_facts(
    session_dir: &Path,
    generation: &str,
) -> Option<(SnapshotModel, Option<String>)> {
    let path = session_dir
        .join("generations")
        .join(generation)
        .join("runtime-state.json");
    let state: Value = serde_json::from_reader(File::open(path).ok()?).ok()?;
    let session_metadata = state.pointer("/session_metadata")?.as_object()?;
    let model = session_metadata
        .get("active_model")
        .and_then(Value::as_str)
        .filter(|model| !model.is_empty())
        .map(catalog_model_id);
    let thinking = session_metadata
        .get("reasoning_effort")
        .and_then(Value::as_str)
        .filter(|level| !level.is_empty())
        .map(str::to_owned);
    Some((SnapshotModel(model), thinking))
}

/// A model alias read from a generation snapshot, kept separate from the
/// metadata alias so the snapshot can fill a missing fact without
/// overriding a direct observation.
struct SnapshotModel(Option<String>);

impl SnapshotModel {
    fn or(self, metadata_model: Option<String>) -> Option<String> {
        self.0.or(metadata_model)
    }
}

/// The journal segment files in sequence order. The segment names are
/// zero-padded, so lexicographic order equals sequence order.
fn journal_files(session_dir: &Path) -> anyhow::Result<Vec<std::path::PathBuf>> {
    let journal_dir = session_dir.join("journal");
    let entries = std::fs::read_dir(&journal_dir)?;
    let mut paths = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let is_jsonl = path
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("jsonl"));
        if is_jsonl {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}

/// The cumulative session usage from the newest projection state.
#[derive(Default)]
struct CumulativeUsage {
    usage: Usage,
    updated_at_ms: Option<i64>,
}

impl CumulativeUsage {
    /// Tracks the newest projection state seen so far. Each projection
    /// state carries the whole session total, so the last one wins.
    fn observe(&mut self, session: &Value) {
        let Some(token_usage) = session.pointer("/tokenUsage") else {
            return;
        };
        let counter = |name: &str| token_usage.get(name).and_then(Value::as_u64).unwrap_or(0);
        self.usage = Usage {
            input_tokens: counter("inputTokens"),
            output_tokens: counter("outputTokens"),
            cache_read_tokens: counter("cachedInputTokens"),
            cache_creation_tokens: 0,
            cache_creation_1h_tokens: 0,
        };
        self.updated_at_ms = session.get("updatedAt").and_then(Value::as_i64);
    }

    /// One session-level usage event. A zero total stays unobserved.
    fn into_event(
        self,
        model: &Option<String>,
        thinking_level: &Option<String>,
    ) -> Option<NormalizedEvent> {
        if self.usage == Usage::default() {
            return None;
        }
        let mut event = NormalizedEvent::new(Role::Assistant);
        event.usage = self.usage;
        event.model = model.clone();
        event.thinking_mode = thinking_level.clone();
        event.ts_ms = self.updated_at_ms;
        Some(event)
    }
}

/// Reads one journal row. Returns `false` when the row does not match the
/// characterized row contract. Tool intents emit one tool event each; the
/// newest projection state tracks the cumulative usage; every other
/// recognized row type carries no retained fact.
fn observe_journal_row(
    value: &Value,
    model: &Option<String>,
    thinking_level: &Option<String>,
    cumulative: &mut CumulativeUsage,
    sink: &mut dyn RecordSink,
) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    let Some(row_type) = object.get("type").and_then(Value::as_str) else {
        return false;
    };
    if !KNOWN_ROW_TYPES.contains(&row_type) || !object.get("payload").is_some_and(Value::is_object)
    {
        return false;
    }
    let payload = &object["payload"];
    match row_type {
        "projection_delta" => {
            let Some(ops) = payload.get("delta").and_then(Value::as_array) else {
                return false;
            };
            for op in ops {
                if let Some(session) = op.pointer("/state/session") {
                    cumulative.observe(session);
                }
            }
            true
        }
        "action_intent" => {
            if payload.get("kind").and_then(Value::as_str) != Some("tool") {
                return true;
            }
            let Some(name) = payload
                .pointer("/request/call/name")
                .and_then(Value::as_str)
                .filter(|name| !name.is_empty())
            else {
                return false;
            };
            let mut event = NormalizedEvent::new(Role::Assistant);
            event.tools = vec![ToolCall {
                name: name.to_owned(),
                category: ToolCategory::Other,
                detail: None,
            }];
            event.model = model.clone();
            event.thinking_mode = thinking_level.clone();
            sink.record(crate::analysis::NormalizedRecord::MetricsEvent(Box::new(
                event,
            )));
            true
        }
        _ => true,
    }
}

/// Maps a Mistral Vibe model alias to its models.dev catalog id.
/// Mistral Vibe records GLM models with a hyphenated alias (`glm-5-3`)
/// while the catalog lists them under the Mistral provider as `zai-glm-5-3`.
/// Unrecognized aliases are returned verbatim so native Mistral model ids
/// keep resolving through their existing catalog keys.
fn catalog_model_id(active_model: &str) -> String {
    let trimmed = active_model.trim();
    let lower = trimmed.to_lowercase();
    if lower.strip_prefix("glm-").is_some_and(|rest| {
        rest.len() >= 3
            && rest
                .split('-')
                .all(|part| !part.is_empty() && part.chars().all(|c| c.is_ascii_digit()))
    }) {
        return format!("zai-{lower}");
    }
    trimmed.to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::interface::RawSource;
    use tempfile::TempDir;

    const META: &str = r#"{
        "session_id": "11111111-1111-4111-8111-111111111111",
        "start_time": "2026-01-01T00:00:00+00:00",
        "end_time": "2026-01-01T00:10:00+00:00",
        "environment": {"working_directory": "/tmp/synthetic"},
        "config": null,
        "title": null,
        "title_source": "auto",
        "child_sessions": [],
        "parent_session_id": null
    }"#;

    const CURRENT: &str = r#"{
        "generation": "0000000000000012",
        "manifest_sha256": "6023e0ec3f5d7bb15ea920f206861ba2a7aef685af8fd9e260947a382779e039",
        "session_id": "11111111-1111-4111-8111-111111111111",
        "snapshot_sequence": 12,
        "store_format": "mistral.vibe.unified-session-store/v1",
        "store_format_minor": 7
    }"#;

    const RUNTIME_STATE: &str = r#"{
        "identity": {
            "depth": 0,
            "kind": "root",
            "parent_session_id": null,
            "root_session_id": "11111111-1111-4111-8111-111111111111",
            "session_id": "11111111-1111-4111-8111-111111111111"
        },
        "session_metadata": {
            "active_model": "mistral-large-latest",
            "agent_name": "auto-approve",
            "cwd": "/tmp/synthetic",
            "parent_session_id": null,
            "reasoning_effort": "high"
        }
    }"#;

    const JOURNAL: &str = concat!(
        r#"{"payload":{"checkpoint_capabilities":{"agent_types":[],"skills":[],"tool_groups":[]}},"sequence":1,"type":"core_input"}"#,
        "\n",
        r#"{"payload":{"delta":[{"op":"upsert","state":{"session":{"createdAt":1790000000000,"id":"11111111-1111-4111-8111-111111111111","tokenUsage":{"cachedInputTokens":900,"inputTokens":1000,"outputTokens":40,"totalTokens":1040},"updatedAt":1790000600000}}}]},"sequence":2,"type":"projection_delta"}"#,
        "\n",
        r#"{"payload":{"action_id":"a","kind":"tool","recovery_mode":"fail","request":{"action_id":"a","call":{"arguments":{"command":"ls"},"name":"file_system.bash","type":"runtime_builtin"},"call_id":"call-1","turn_id":"turn-1","type":"runtime_builtin_tool_call"}},"sequence":3,"type":"action_intent"}"#,
        "\n",
        r#"{"payload":{"action_id":"a","kind":"tool","result":{"action_id":"a","call_id":"call-1","result":{"content":[],"type":"success"},"type":"tool_succeeded"},"state":"succeeded"},"sequence":4,"type":"action_result"}"#,
        "\n",
        r#"{"payload":{"callback_id":"c1","kind":"approval"},"sequence":5,"type":"callback_registered"}"#,
        "\n"
    );

    fn store(temp: &TempDir) -> SessionInput {
        let dir = temp.path().join("11111111-1111-4111-8111-111111111111");
        std::fs::create_dir_all(dir.join("journal")).unwrap();
        std::fs::create_dir_all(dir.join("generations").join("0000000000000012")).unwrap();
        std::fs::write(dir.join("meta.json"), META).unwrap();
        std::fs::write(dir.join("CURRENT"), CURRENT).unwrap();
        std::fs::write(
            dir.join("generations")
                .join("0000000000000012")
                .join("runtime-state.json"),
            RUNTIME_STATE,
        )
        .unwrap();
        std::fs::write(dir.join("journal").join("0000000000000001.jsonl"), JOURNAL).unwrap();
        SessionInput {
            agent: "mistral-vibe".to_owned(),
            session_id: "11111111-1111-4111-8111-111111111111".to_owned(),
            source: RawSource::MistralVibeUnifiedBundle { session_dir: dir },
            source_format: SourceFormat::MistralVibeUnifiedStoreV1,
            fork_parent_session_id: None,
        }
    }

    #[test]
    fn store_normalizes_model_usage_and_tools() {
        let temp = TempDir::new().unwrap();
        let session = MistralVibeSessionReader.normalize(&store(&temp)).unwrap();
        assert!(
            session
                .events
                .iter()
                .any(|event| event.model.as_deref() == Some("mistral-large-latest"))
        );
        let usage = session
            .events
            .iter()
            .find(|event| event.usage.input_tokens > 0)
            .expect("cumulative usage event");
        assert_eq!(usage.usage.input_tokens, 1000);
        assert_eq!(usage.usage.output_tokens, 40);
        assert_eq!(usage.usage.cache_read_tokens, 900);
        assert_eq!(usage.thinking_mode.as_deref(), Some("high"));
        assert!(session.events.iter().any(|event| {
            event
                .tools
                .iter()
                .any(|tool| tool.name == "file_system.bash")
        }));
    }

    #[test]
    fn glm_alias_maps_to_catalog_id() {
        let temp = TempDir::new().unwrap();
        let input = store(&temp);
        let RawSource::MistralVibeUnifiedBundle { session_dir } = input.source.clone() else {
            unreachable!()
        };
        std::fs::write(
            session_dir
                .join("generations")
                .join("0000000000000012")
                .join("runtime-state.json"),
            RUNTIME_STATE.replace("mistral-large-latest", "glm-5-3"),
        )
        .unwrap();
        let session = MistralVibeSessionReader.normalize(&input).unwrap();
        assert!(
            session
                .events
                .iter()
                .any(|event| event.model.as_deref() == Some("zai-glm-5-3"))
        );
    }

    #[test]
    fn catalog_model_id_preserves_other_aliases() {
        assert_eq!(
            catalog_model_id("mistral-large-latest"),
            "mistral-large-latest"
        );
        assert_eq!(catalog_model_id("glm-5-3"), "zai-glm-5-3");
        assert_eq!(catalog_model_id("GLM-5-3 "), "zai-glm-5-3");
        assert_eq!(catalog_model_id("glm-52-10"), "zai-glm-52-10");
        assert_eq!(catalog_model_id("glm-x"), "glm-x");
        assert_eq!(catalog_model_id("glm--5"), "glm--5");
        assert_eq!(catalog_model_id("glm"), "glm");
        assert_eq!(catalog_model_id("mistral/glm-5-3"), "mistral/glm-5-3");
    }

    #[test]
    fn capabilities_are_mistral_vibe() {
        let temp = TempDir::new().unwrap();
        let caps = MistralVibeSessionReader.capabilities(&store(&temp));
        assert_eq!(caps.source_format, SourceFormat::MistralVibeUnifiedStoreV1);
        assert!(caps.model_identity);
        assert!(caps.token_classes);
        assert!(caps.reasoning_effort_tier);
        assert!(!caps.subagent_relationships);
        assert!(!caps.request_context_tokens);
        assert!(!caps.compaction_boundaries);
    }

    #[test]
    fn a_wrong_store_format_fails() {
        let temp = TempDir::new().unwrap();
        let input = store(&temp);
        let RawSource::MistralVibeUnifiedBundle { session_dir } = input.source.clone() else {
            unreachable!()
        };
        std::fs::write(
            session_dir.join("CURRENT"),
            CURRENT.replace("mistral.vibe.unified-session-store/v1", "other/v9"),
        )
        .unwrap();
        assert!(MistralVibeSessionReader.normalize(&input).is_err());
    }

    #[test]
    fn a_mismatched_current_session_id_fails() {
        let temp = TempDir::new().unwrap();
        let input = store(&temp);
        let RawSource::MistralVibeUnifiedBundle { session_dir } = input.source.clone() else {
            unreachable!()
        };
        std::fs::write(
            session_dir.join("CURRENT"),
            CURRENT.replace(
                "11111111-1111-4111-8111-111111111111",
                "22222222-2222-4222-8222-222222222222",
            ),
        )
        .unwrap();
        assert!(MistralVibeSessionReader.normalize(&input).is_err());
    }

    #[test]
    fn an_unknown_journal_row_marks_the_source_partial() {
        let temp = TempDir::new().unwrap();
        let input = store(&temp);
        let RawSource::MistralVibeUnifiedBundle { session_dir } = input.source.clone() else {
            unreachable!()
        };
        std::fs::write(
            session_dir.join("journal").join("0000000000000002.jsonl"),
            concat!(
                r#"{"payload":{"future_field":true},"sequence":6,"type":"future_row"}"#,
                "\n"
            ),
        )
        .unwrap();
        let session = MistralVibeSessionReader.normalize(&input).unwrap();
        assert!(
            session
                .events
                .iter()
                .any(|event| event.usage.input_tokens == 1000)
        );
    }

    #[test]
    fn a_store_without_a_snapshot_keeps_model_unknown() {
        let temp = TempDir::new().unwrap();
        let input = store(&temp);
        let RawSource::MistralVibeUnifiedBundle { session_dir } = input.source.clone() else {
            unreachable!()
        };
        std::fs::remove_file(
            session_dir
                .join("generations")
                .join("0000000000000012")
                .join("runtime-state.json"),
        )
        .unwrap();
        let session = MistralVibeSessionReader.normalize(&input).unwrap();
        assert!(session.events.iter().all(|event| event.model.is_none()));
        assert!(
            session
                .events
                .iter()
                .any(|event| event.usage.input_tokens == 1000)
        );
    }
}
