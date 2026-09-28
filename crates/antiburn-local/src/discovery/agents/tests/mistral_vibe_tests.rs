use super::*;
use tempfile::TempDir;

fn unified_sessions_dir(home: &Path) -> PathBuf {
    home.join(".vibe")
        .join("logs")
        .join("session")
        .join("unified")
}

const SID: &str = "11111111-1111-4111-8111-111111111111";

/// Runs `body` with `HOME` pointed at `home` and `vars` applied, then
/// restores the previous environment.
fn with_env<T>(home: &Path, vars: &[(&str, &Path)], body: impl FnOnce() -> T) -> T {
    let mut previous = vec![("HOME", std::env::var_os("HOME"))];
    // SAFETY: every caller is serialised, so no other test thread can
    // observe the partial environment.
    unsafe { std::env::set_var("HOME", home) };
    for (key, value) in vars {
        previous.push((key, std::env::var_os(key)));
        unsafe { std::env::set_var(key, value) };
    }
    let result = body();
    for (key, value) in previous.into_iter().rev() {
        unsafe {
            match value {
                Some(value) => std::env::set_var(key, value),
                None => std::env::remove_var(key),
            }
        }
    }
    result
}

fn write_store(unified_dir: &Path, session_id: &str) -> PathBuf {
    let session_dir = unified_dir.join(session_id);
    std::fs::create_dir_all(session_dir.join("journal")).unwrap();
    let meta_path = session_dir.join("meta.json");
    std::fs::write(
        &meta_path,
        format!(r#"{{"session_id":"{session_id}","environment":{{"working_directory":"/tmp/synthetic"}}}}"#),
    )
    .unwrap();
    std::fs::write(
        session_dir.join("CURRENT"),
        format!(r#"{{"session_id":"{session_id}","store_format":"mistral.vibe.unified-session-store/v1","generation":"0000000000000001","store_format_minor":7}}"#),
    )
    .unwrap();
    std::fs::write(
        session_dir.join("journal").join("0000000000000001.jsonl"),
        "{}\n",
    )
    .unwrap();
    meta_path
}

#[test]
fn vibe_home_defaults_to_the_dotdir() {
    // `env_path_when_real_home` reads the process environment only for the
    // real home directory, so a synthetic home always resolves the default.
    let home = Path::new("/synthetic/home");
    assert_eq!(vibe_home_in(home), home.join(".vibe"));
}

#[test]
#[serial_test::serial]
fn vibe_home_replaces_the_root() {
    let home = TempDir::new().unwrap();
    let moved = home.path().join("vibe-alt");
    let root = with_env(home.path(), &[("VIBE_HOME", &moved)], || {
        vibe_home_in(home.path())
    });
    assert_eq!(root, moved);
}

#[test]
#[serial_test::serial]
fn an_empty_vibe_home_keeps_the_default() {
    let home = TempDir::new().unwrap();
    let empty = Path::new("");
    let root = with_env(home.path(), &[("VIBE_HOME", empty)], || {
        vibe_home_in(home.path())
    });
    assert_eq!(root, home.path().join(".vibe"));
}

#[test]
#[serial_test::serial]
fn test_discover_recent_finds_unified_meta_json() {
    let home = TempDir::new().unwrap();
    let meta_path = write_store(&unified_sessions_dir(home.path()), SID);

    let logs = with_env(home.path(), &[], || {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("test runtime")
            .block_on(MistralVibeExplorer.discover_recent(i64::MAX / 2, i64::MAX / 2))
    });

    assert_eq!(logs.len(), 1);
    assert!(matches!(
        &logs[0].source,
        SessionSource::File(path) if *path == meta_path
    ));
    assert_eq!(logs[0].agent_type, AgentKind::MistralVibe);
    assert!(logs[0].updated_at.unwrap() > 0);
}

#[test]
#[serial_test::serial]
fn test_discover_recent_skips_child_stores() {
    let home = TempDir::new().unwrap();
    let unified = unified_sessions_dir(home.path());
    write_store(&unified, SID);
    write_store(&unified, "child-38edb7d3506726c8d0017882");

    let logs = with_env(home.path(), &[], || {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("test runtime")
            .block_on(MistralVibeExplorer.discover_recent(i64::MAX / 2, i64::MAX / 2))
    });

    assert_eq!(logs.len(), 1);
    assert!(matches!(
        &logs[0].source,
        SessionSource::File(path) if path.parent().is_some_and(|dir| dir.ends_with(SID))
    ));
}

#[test]
#[serial_test::serial]
fn test_discover_recent_skips_stores_without_meta() {
    let home = TempDir::new().unwrap();
    let session_dir = unified_sessions_dir(home.path()).join(SID);
    std::fs::create_dir_all(session_dir.join("journal")).unwrap();
    std::fs::write(
        session_dir.join("journal").join("0000000000000001.jsonl"),
        "{}\n",
    )
    .unwrap();

    let logs = with_env(home.path(), &[], || {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("test runtime")
            .block_on(MistralVibeExplorer.discover_recent(i64::MAX / 2, i64::MAX / 2))
    });

    assert!(logs.is_empty());
}

#[test]
#[serial_test::serial]
fn test_discover_recent_graceful_when_missing() {
    let home = TempDir::new().unwrap();
    let logs = with_env(home.path(), &[], || {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("test runtime")
            .block_on(MistralVibeExplorer.discover_recent(i64::MAX / 2, i64::MAX / 2))
    });

    assert!(logs.is_empty());
}

#[test]
#[serial_test::serial]
fn test_direct_session_source_matches_the_directory_id() {
    let home = TempDir::new().unwrap();
    write_store(&unified_sessions_dir(home.path()), SID);

    let source = with_env(home.path(), &[], || {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("test runtime")
            .block_on(MistralVibeExplorer.direct_session_source(SID))
    });

    assert!(matches!(
        source,
        DirectSessionSource::Found(SessionSource::File(path)) if path.ends_with("meta.json")
    ));
}

#[test]
#[serial_test::serial]
fn test_direct_session_source_rejects_a_child_store() {
    let home = TempDir::new().unwrap();
    let unified = unified_sessions_dir(home.path());
    write_store(&unified, SID);
    write_store(&unified, "child-38edb7d3506726c8d0017882");

    let child = with_env(home.path(), &[], || {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("test runtime")
            .block_on(MistralVibeExplorer.direct_session_source("child-38edb7d3506726c8d0017882"))
    });
    let unknown = with_env(home.path(), &[], || {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("test runtime")
            .block_on(
                MistralVibeExplorer.direct_session_source("99999999-9999-4999-8999-999999999999"),
            )
    });

    assert!(matches!(child, DirectSessionSource::Missing));
    assert!(matches!(unknown, DirectSessionSource::Missing));
}

#[tokio::test]
async fn test_owns_path_matches_unified_session_path() {
    assert!(MistralVibeExplorer.owns_path(
        "/users/foo/.vibe/logs/session/unified/09f22d1e-3633-71f6-6f91-823b1291df14/meta.json"
    ));
}

#[tokio::test]
async fn test_owns_path_rejects_other_roots() {
    assert!(
        !MistralVibeExplorer.owns_path(
            "/users/foo/.vibe/logs/session/session_20260526_010203_0a1b2c3d4e5f/meta.json"
        )
    );
    assert!(!MistralVibeExplorer.owns_path("/users/foo/.vibe/logs/vibe.log"));
    assert!(!MistralVibeExplorer.owns_path("/users/foo/.pi/agent/sessions/proj/a.jsonl"));
}

#[tokio::test]
async fn test_recover_session_id_extracts_the_directory_name() {
    let path = Path::new(
        "/home/foo/.vibe/logs/session/unified/09f22d1e-3633-71f6-6f91-823b1291df14/meta.json",
    );
    assert_eq!(
        MistralVibeExplorer
            .recover_session_id_from_path(path)
            .as_deref(),
        Some("09f22d1e-3633-71f6-6f91-823b1291df14")
    );
}

#[tokio::test]
async fn test_recover_session_id_rejects_child_and_stray_paths() {
    let child =
        Path::new("/home/foo/.vibe/logs/session/unified/child-38edb7d3506726c8d0017882/meta.json");
    assert_eq!(
        MistralVibeExplorer.recover_session_id_from_path(child),
        None
    );
    let stray = Path::new("/home/foo/.vibe/logs/session/unified.json");
    assert_eq!(
        MistralVibeExplorer.recover_session_id_from_path(stray),
        None
    );
}
