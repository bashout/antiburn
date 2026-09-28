//! Mistral Vibe agent log discovery.
//!
//! Mistral Vibe stores unified sessions at
//! `~/.vibe/logs/session/unified/`. Each root session is a directory named
//! by its full session id that holds a `meta.json` metadata file beside
//! the store files (`CURRENT`, `journal/`, and `generations/`). `VIBE_HOME`
//! moves the root. The `session_logging.save_dir` config key can move it
//! again; that root is not discovered and see `docs/session-coverage.md`.
//!
//! Subagent stores are `child-*` directories beside the root sessions.
//! They hold no `meta.json`, so they are not session sources. The explorer
//! reports a root session's `meta.json` as the session source, and the
//! newest store file mtime tracks an active session.

use std::path::{Path, PathBuf};

use crate::discovery::scanner::AgentKind;
use crate::discovery::{
    AgentExplorer, DirectSessionSource, SessionLog, SessionSource, SurfacePaths, WatchRoot,
    env_path_when_real_home, home_dir,
};
use async_trait::async_trait;

pub struct MistralVibeExplorer;

#[async_trait]
impl AgentExplorer for MistralVibeExplorer {
    async fn discover_recent(&self, now: i64, since_secs: i64) -> Vec<SessionLog> {
        let Some(home) = home_dir() else {
            return Vec::new();
        };
        let cutoff = now - since_secs;
        session_stores_in(&home)
            .await
            .into_iter()
            .filter(|store| store.activity_epoch >= cutoff)
            .map(|store| SessionLog {
                agent_type: AgentKind::MistralVibe,
                source: SessionSource::File(store.meta_path),
                updated_at: Some(store.activity_epoch),
                environment: Default::default(),
            })
            .collect()
    }

    /// Owns the Mistral Vibe unified session tree at
    /// `~/.vibe/logs/session/unified/**`. Single substring covers macOS,
    /// Linux, and Windows under `%USERPROFILE%/.vibe/` after path
    /// normalisation.
    fn owns_path(&self, path_lower: &str) -> bool {
        path_lower.contains("/.vibe/logs/session/unified/")
    }

    fn unmatched_surface(&self) -> &'static str {
        "cli"
    }

    /// CLI-only: `~/.vibe/logs/session/unified/`. Mistral Vibe has no IDE
    /// companion. `VIBE_HOME` replaces the `.vibe` dotdir root.
    fn surface_paths(&self, home: &Path) -> SurfacePaths {
        SurfacePaths {
            cli: vec![unified_dir_in(home)],
            ide_desktop: Vec::new(),
            mirror: Vec::new(),
        }
    }

    fn watch_roots(&self, home: &Path) -> Vec<WatchRoot> {
        self.surface_paths(home)
            .cli
            .into_iter()
            .map(WatchRoot::recursive)
            .collect()
    }

    // Title lookup: inherits the `Scan` default. The metadata file sits
    // beside the store, so a Direct point query is no cheaper than the
    // batched scan path.

    /// Point query: a root session directory is named by the full session
    /// id, so the store resolves through one directory name. `meta.json`
    /// confirms the id, because a directory name alone does not prove a
    /// session store. A miss returns `Missing`: subagent stores hold no
    /// metadata file, so the caller has no fallback to try.
    async fn direct_session_source(&self, session_id: &str) -> DirectSessionSource {
        if session_id.is_empty() || session_id.starts_with("child-") {
            return DirectSessionSource::Missing;
        }
        let Some(home) = home_dir() else {
            return DirectSessionSource::Missing;
        };
        let meta_path = unified_dir_in(&home).join(session_id).join("meta.json");
        let Ok(content) = tokio::fs::read_to_string(&meta_path).await else {
            return DirectSessionSource::Missing;
        };
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&content) else {
            return DirectSessionSource::Missing;
        };
        if value.get("session_id").and_then(|id| id.as_str()) == Some(session_id) {
            DirectSessionSource::Found(SessionSource::File(meta_path))
        } else {
            DirectSessionSource::Missing
        }
    }

    /// The session directory is named by the full session id, so the id is
    /// the parent directory name of the reported metadata file. A
    /// `child-*` directory is a subagent store, not a session source, and
    /// returns `None` rather than guessing.
    fn recover_session_id_from_path(&self, file: &Path) -> Option<String> {
        let dir = file.parent()?;
        let name = dir.file_name()?.to_str()?;
        if name.starts_with("child-")
            || dir.parent()?.file_name()?.to_str()? != "unified"
            || name.is_empty()
        {
            return None;
        }
        Some(String::from(name))
    }
}

/// A discovered root session store.
struct SessionStore {
    meta_path: PathBuf,
    /// The newest store file mtime: the metadata file, `CURRENT`, or a
    /// journal segment. An active session appends journal rows between
    /// metadata rewrites, so the journal decides recency.
    activity_epoch: i64,
}

/// The root session stores under the unified tree. Each one holds a
/// `meta.json` metadata file; `child-*` directories are subagent stores
/// and stay excluded.
async fn session_stores_in(home: &Path) -> Vec<SessionStore> {
    let unified_dir = unified_dir_in(home);
    let mut entries = match tokio::fs::read_dir(&unified_dir).await {
        Ok(entries) => entries,
        Err(_) => return Vec::new(),
    };
    let mut stores = Vec::new();
    while let Ok(Some(entry)) = entries.next_entry().await {
        let Ok(file_type) = entry.file_type().await else {
            continue;
        };
        if !file_type.is_dir() {
            continue;
        }
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        if name.starts_with("child-") {
            continue;
        }
        let session_dir = entry.path();
        let meta_path = session_dir.join("meta.json");
        if tokio::fs::metadata(&meta_path).await.is_err() {
            continue;
        }
        let activity_epoch = newest_store_mtime(&session_dir, &meta_path);
        stores.push(SessionStore {
            meta_path,
            activity_epoch,
        });
    }
    stores
}

/// The newest mtime among the metadata file, `CURRENT`, and the journal
/// segments. One bounded directory read keeps the cost per session flat.
fn newest_store_mtime(session_dir: &Path, meta_path: &Path) -> i64 {
    let mtime = |path: &Path| {
        std::fs::metadata(path)
            .and_then(|metadata| metadata.modified())
            .ok()
            .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|duration| duration.as_secs() as i64)
            .unwrap_or(0)
    };
    let mut newest = mtime(meta_path);
    newest = newest.max(mtime(&session_dir.join("CURRENT")));
    if let Ok(entries) = std::fs::read_dir(session_dir.join("journal")) {
        for entry in entries.flatten() {
            newest = newest.max(mtime(&entry.path()));
        }
    }
    newest
}

#[cfg(test)]
pub(crate) fn sample_log_path(home: &Path) -> PathBuf {
    unified_dir_in(home)
        .join("11111111-1111-4111-8111-111111111111")
        .join("meta.json")
}

/// The Mistral Vibe home under `home`.
///
/// `VIBE_HOME` replaces the whole `.vibe` root. It can name any directory,
/// so unlike the OMP config rename it does not have to stay under `home`.
/// An empty value keeps the default.
pub(crate) fn vibe_home_in(home: &Path) -> PathBuf {
    env_path_when_real_home(home, "VIBE_HOME")
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| home.join(".vibe"))
}

/// The unified session store directory under `home`.
fn unified_dir_in(home: &Path) -> PathBuf {
    vibe_home_in(home)
        .join("logs")
        .join("session")
        .join("unified")
}

#[cfg(test)]
#[path = "tests/mistral_vibe_tests.rs"]
mod tests;
