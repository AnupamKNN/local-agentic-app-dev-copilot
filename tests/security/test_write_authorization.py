"""Isolated security regression tests for the filesystem MCP server."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERVER_PATH = (
    PROJECT_ROOT
    / "research_notebooks"
    / "agents"
    / "_mcp_sandbox_server.py"
)

DEFAULT_SCOPE = {
    "role": None,
    "task_id": None,
    "allowed_creations": [],
    "allowed_overwrites": [],
}


@pytest.fixture(scope="module")
def sandbox_server_module():
    """Load the real server module without starting its MCP server."""
    if not SERVER_PATH.is_file():
        pytest.fail(f"Sandbox server source not found: {SERVER_PATH}")

    module_name = "_copilot_sandbox_server_regression_tests"
    spec = importlib.util.spec_from_file_location(module_name, SERVER_PATH)

    if spec is None or spec.loader is None:
        pytest.fail("Could not create an import spec for the sandbox server.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(module_name, None)


@pytest.fixture
def isolated_state(tmp_path, monkeypatch, sandbox_server_module):
    """Redirect server state to a fresh temporary sandbox and manifest."""
    sandbox = tmp_path / "mcp_sandbox"
    runtime = tmp_path / ".copilot_runtime"
    sandbox.mkdir()
    runtime.mkdir()

    manifest_path = runtime / "active_write_scope.json"
    manifest_path.write_text(
        json.dumps(DEFAULT_SCOPE, indent=2),
        encoding="utf-8",
    )

    # Never use the real project sandbox or live authorization manifest.
    monkeypatch.setattr(sandbox_server_module, "MCP_SANDBOX", sandbox.resolve())
    monkeypatch.setattr(
        sandbox_server_module,
        "WRITE_SCOPE_FILE",
        manifest_path,
    )

    return {
        "server": sandbox_server_module,
        "sandbox": sandbox,
        "manifest": manifest_path,
    }


def save_manifest(manifest_path: Path, scope: dict) -> None:
    manifest_path.write_text(
        json.dumps(scope, indent=2),
        encoding="utf-8",
    )


def load_manifest(manifest_path: Path) -> dict:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def creation_scope(path: str, content: str) -> dict:
    return {
        "role": "implementer",
        "task_id": "regression-test-task",
        "allowed_creations": [
            {
                "path": path,
                "approved_content_sha256": sha256_text(content),
            }
        ],
        "allowed_overwrites": [],
    }


def overwrite_scope(
    path: str,
    existing_content: str,
    proposed_content: str,
) -> dict:
    return {
        "role": "implementer",
        "task_id": "regression-test-task",
        "allowed_creations": [],
        "allowed_overwrites": [
            {
                "path": path,
                "expected_existing_sha256": sha256_text(existing_content),
                "approved_content_sha256": sha256_text(proposed_content),
            }
        ],
    }


def test_default_deny_rejects_write(isolated_state):
    server = isolated_state["server"]
    target = isolated_state["sandbox"] / "probe.txt"

    with pytest.raises(PermissionError, match="no valid one-time grant"):
        server.write_file("probe.txt", "unauthorized")

    assert not target.exists()
    assert load_manifest(isolated_state["manifest"]) == DEFAULT_SCOPE


def test_missing_manifest_denies_authorization(isolated_state):
    server = isolated_state["server"]
    isolated_state["manifest"].unlink()

    with pytest.raises(PermissionError, match="missing or invalid"):
        server._load_write_scope()


def test_legacy_path_only_manifest_is_rejected(isolated_state):
    manifest_path = isolated_state["manifest"]
    save_manifest(
        manifest_path,
        {
            "role": "implementer",
            "task_id": "legacy-task",
            "allowed_paths": ["probe.txt"],
        },
    )

    with pytest.raises(PermissionError, match="schema"):
        isolated_state["server"]._load_write_scope()


def test_invalid_creation_hash_is_rejected(isolated_state):
    save_manifest(
        isolated_state["manifest"],
        {
            "role": "implementer",
            "task_id": "regression-test-task",
            "allowed_creations": [
                {
                    "path": "probe.txt",
                    "approved_content_sha256": "z" * 64,
                }
            ],
            "allowed_overwrites": [],
        },
    )

    with pytest.raises(PermissionError, match="invalid creation content hash"):
        isolated_state["server"]._load_write_scope()


def test_duplicate_or_conflicting_path_grants_are_rejected(isolated_state):
    path = "probe.txt"
    save_manifest(
        isolated_state["manifest"],
        {
            "role": "implementer",
            "task_id": "regression-test-task",
            "allowed_creations": [
                {
                    "path": path,
                    "approved_content_sha256": sha256_text("first"),
                },
                {
                    "path": path,
                    "approved_content_sha256": sha256_text("second"),
                },
            ],
            "allowed_overwrites": [],
        },
    )

    with pytest.raises(PermissionError, match="duplicate or conflicting"):
        isolated_state["server"]._load_write_scope()


def test_content_hash_mismatch_is_denied_without_consuming_grant(
    isolated_state,
):
    path = "probe.txt"
    approved = "approved content"
    attempted = "different content"
    manifest_path = isolated_state["manifest"]
    target = isolated_state["sandbox"] / path

    scope = creation_scope(path, approved)
    save_manifest(manifest_path, scope)

    with pytest.raises(PermissionError, match="proposed content was not approved"):
        isolated_state["server"].write_file(path, attempted)

    assert not target.exists()
    assert load_manifest(manifest_path) == scope


def test_creation_grant_cannot_overwrite_existing_file(isolated_state):
    path = "probe.txt"
    original = "original contents"
    proposed = "replacement contents"
    target = isolated_state["sandbox"] / path
    target.write_text(original, encoding="utf-8")

    scope = creation_scope(path, proposed)
    save_manifest(isolated_state["manifest"], scope)

    with pytest.raises(PermissionError, match="cannot overwrite"):
        isolated_state["server"].write_file(path, proposed)

    assert target.read_text(encoding="utf-8") == original
    assert load_manifest(isolated_state["manifest"]) == scope


def test_exact_creation_succeeds_and_consumes_one_time_grant(isolated_state):
    path = "probe.txt"
    content = "approved bytes\n"
    target = isolated_state["sandbox"] / path
    manifest_path = isolated_state["manifest"]

    save_manifest(manifest_path, creation_scope(path, content))

    result = isolated_state["server"].write_file(path, content)

    assert result == f"Successfully wrote {len(content)} characters to {path}"
    assert target.read_bytes() == content.encode("utf-8")

    consumed = load_manifest(manifest_path)
    assert consumed["role"] == "implementer"
    assert consumed["task_id"] == "regression-test-task"
    assert consumed["allowed_creations"] == []
    assert consumed["allowed_overwrites"] == []

    # Reusing the consumed grant must fail.
    with pytest.raises(PermissionError, match="no valid one-time grant"):
        isolated_state["server"].write_file(path, content)

    assert target.read_bytes() == content.encode("utf-8")
    assert load_manifest(manifest_path) == consumed


def test_overwrite_rejects_changed_existing_content(isolated_state):
    path = "probe.txt"
    initial = "original contents"
    changed = "unexpected contents"
    proposed = "approved replacement"
    target = isolated_state["sandbox"] / path
    target.write_text(changed, encoding="utf-8")

    scope = overwrite_scope(path, initial, proposed)
    save_manifest(isolated_state["manifest"], scope)

    with pytest.raises(PermissionError, match="existing content hash changed"):
        isolated_state["server"].write_file(path, proposed)

    assert target.read_text(encoding="utf-8") == changed
    assert load_manifest(isolated_state["manifest"]) == scope


def test_exact_overwrite_succeeds_and_consumes_grant(isolated_state):
    path = "probe.txt"
    initial = "initial contents"
    proposed = "approved replacement"
    target = isolated_state["sandbox"] / path
    manifest_path = isolated_state["manifest"]
    target.write_text(initial, encoding="utf-8")

    save_manifest(manifest_path, overwrite_scope(path, initial, proposed))

    result = isolated_state["server"].write_file(path, proposed)

    assert result == f"Successfully wrote {len(proposed)} characters to {path}"
    assert target.read_bytes() == proposed.encode("utf-8")

    consumed = load_manifest(manifest_path)
    assert consumed["allowed_overwrites"] == []
    assert consumed["allowed_creations"] == []


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "../outside.txt",
        "../../outside.txt",
        "/tmp/outside.txt",
    ],
)
def test_path_resolution_rejects_paths_outside_sandbox(
    isolated_state,
    unsafe_path,
):
    with pytest.raises(ValueError, match="escapes the MCP sandbox"):
        isolated_state["server"].resolve_sandbox_path(unsafe_path)


def test_symlink_escape_is_rejected_when_supported(isolated_state, tmp_path):
    server = isolated_state["server"]
    sandbox = isolated_state["sandbox"]
    outside = tmp_path / "outside"
    outside.mkdir()
    link = sandbox / "escape"

    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Directory symlink is unavailable in this environment: {exc}")

    with pytest.raises(ValueError, match="escapes the MCP sandbox"):
        server.resolve_sandbox_path("escape/created.txt")
def test_delete_file_is_denied_by_default(isolated_state):
    """The MCP server must not delete files without deletion authorization."""
    server = isolated_state["server"]
    target = isolated_state["sandbox"] / "protected.txt"
    manifest_path = isolated_state["manifest"]

    original_content = "preserve this file"
    target.write_text(original_content, encoding="utf-8")

    # No role is allowed to delete files. Deny deletion at the server boundary.
    with pytest.raises(PermissionError, match="Deletion is disabled"):
        server.delete_file("protected.txt")

    assert target.is_file()
    assert target.read_text(encoding="utf-8") == original_content
    assert load_manifest(manifest_path) == DEFAULT_SCOPE
def test_final_newline_difference_fails_exact_content_hash(isolated_state):
    """A one-byte final-newline difference must fail the SHA-256 grant."""
    server = isolated_state["server"]
    sandbox = isolated_state["sandbox"]
    manifest_path = isolated_state["manifest"]

    path = "newline_probe.txt"
    approved_content = "first line\\nsecond line\\n"
    submitted_content = approved_content[:-1]

    assert approved_content.encode("utf-8") != submitted_content.encode("utf-8")
    assert len(approved_content.encode("utf-8")) == (
        len(submitted_content.encode("utf-8")) + 1
    )
    assert approved_content.endswith("\\n")
    assert not submitted_content.endswith("\\n")

    scope = creation_scope(path, approved_content)
    save_manifest(manifest_path, scope)

    target = sandbox / path

    with pytest.raises(
        PermissionError,
        match="proposed content was not approved",
    ):
        server.write_file(path, submitted_content)

    # A rejected near-match must neither create the file nor consume the grant.
    assert not target.exists()
    assert load_manifest(manifest_path) == scope

def test_non_object_manifest_is_rejected(isolated_state):
    manifest_path = isolated_state["manifest"]
    manifest_path.write_text("null", encoding="utf-8")

    with pytest.raises(PermissionError, match="invalid authorization manifest"):
        isolated_state["server"]._load_write_scope()


def test_creation_grants_must_be_a_list(isolated_state):
    scope = dict(DEFAULT_SCOPE)
    scope["allowed_creations"] = {"unexpected": "mapping"}
    save_manifest(isolated_state["manifest"], scope)

    with pytest.raises(PermissionError, match="invalid creation grants"):
        isolated_state["server"]._load_write_scope()


def test_overwrite_grants_must_be_a_list(isolated_state):
    scope = dict(DEFAULT_SCOPE)
    scope["allowed_overwrites"] = {"unexpected": "mapping"}
    save_manifest(isolated_state["manifest"], scope)

    with pytest.raises(PermissionError, match="invalid overwrite grants"):
        isolated_state["server"]._load_write_scope()


def test_non_mapping_creation_grant_is_rejected(isolated_state):
    scope = dict(DEFAULT_SCOPE)
    scope["allowed_creations"] = ["not-a-mapping"]
    save_manifest(isolated_state["manifest"], scope)

    with pytest.raises(PermissionError, match="malformed creation grant"):
        isolated_state["server"]._load_write_scope()


def test_creation_grant_with_extra_fields_is_rejected(isolated_state):
    scope = creation_scope("probe.txt", "approved")
    scope["allowed_creations"][0]["unexpected"] = "extra"
    save_manifest(isolated_state["manifest"], scope)

    with pytest.raises(PermissionError, match="invalid creation grant fields"):
        isolated_state["server"]._load_write_scope()


@pytest.mark.parametrize(
    "invalid_path",
    [
        "../outside.txt",
        " folder.txt",
        "folder.txt ",
        r"folder\\file.txt",
        ".",
    ],
)
def test_invalid_creation_grant_path_is_rejected(isolated_state, invalid_path):
    scope = creation_scope(invalid_path, "approved")
    save_manifest(isolated_state["manifest"], scope)

    with pytest.raises(PermissionError, match="invalid creation grant path"):
        isolated_state["server"]._load_write_scope()


def test_invalid_creation_content_hash_is_rejected(isolated_state):
    scope = creation_scope("probe.txt", "approved")
    scope["allowed_creations"][0]["approved_content_sha256"] = "A" * 64
    save_manifest(isolated_state["manifest"], scope)

    with pytest.raises(
        PermissionError,
        match="invalid creation content hash",
    ):
        isolated_state["server"]._load_write_scope()


def test_non_mapping_overwrite_grant_is_rejected(isolated_state):
    scope = dict(DEFAULT_SCOPE)
    scope["allowed_overwrites"] = ["not-a-mapping"]
    save_manifest(isolated_state["manifest"], scope)

    with pytest.raises(PermissionError, match="malformed overwrite grant"):
        isolated_state["server"]._load_write_scope()


def test_overwrite_grant_with_extra_fields_is_rejected(isolated_state):
    scope = overwrite_scope("probe.txt", "existing", "proposed")
    scope["allowed_overwrites"][0]["unexpected"] = "extra"
    save_manifest(isolated_state["manifest"], scope)

    with pytest.raises(PermissionError, match="invalid overwrite grant fields"):
        isolated_state["server"]._load_write_scope()


def test_invalid_overwrite_grant_path_is_rejected(isolated_state):
    scope = overwrite_scope("../outside.txt", "existing", "proposed")
    save_manifest(isolated_state["manifest"], scope)

    with pytest.raises(PermissionError, match="invalid overwrite grant path"):
        isolated_state["server"]._load_write_scope()


def test_invalid_expected_existing_hash_is_rejected(isolated_state):
    scope = overwrite_scope("probe.txt", "existing", "proposed")
    scope["allowed_overwrites"][0]["expected_existing_sha256"] = "bad-hash"
    save_manifest(isolated_state["manifest"], scope)

    with pytest.raises(PermissionError, match="invalid existing content hash"):
        isolated_state["server"]._load_write_scope()


def test_invalid_overwrite_approved_hash_is_rejected(isolated_state):
    scope = overwrite_scope("probe.txt", "existing", "proposed")
    scope["allowed_overwrites"][0]["approved_content_sha256"] = "g" * 64
    save_manifest(isolated_state["manifest"], scope)

    with pytest.raises(PermissionError, match="invalid approved content hash"):
        isolated_state["server"]._load_write_scope()


def test_non_implementer_role_is_rejected_during_manifest_validation(
    isolated_state,
):
    scope = creation_scope("probe.txt", "approved")
    scope["role"] = "reviewer"
    save_manifest(isolated_state["manifest"], scope)

    with pytest.raises(PermissionError, match="only the implementer role"):
        isolated_state["server"]._load_write_scope()


@pytest.mark.parametrize(
    "invalid_task_id",
    [
        None,
        "",
        "   ",
        "x" * 129,
        "bad\nid",
    ],
)
def test_invalid_task_identity_is_rejected(isolated_state, invalid_task_id):
    scope = creation_scope("probe.txt", "approved")
    scope["task_id"] = invalid_task_id
    save_manifest(isolated_state["manifest"], scope)

    with pytest.raises(PermissionError, match="valid role/task identity"):
        isolated_state["server"]._load_write_scope()


def test_overwrite_grant_requires_existing_file(isolated_state):
    path = "missing.txt"
    scope = overwrite_scope(path, "old contents", "new contents")
    save_manifest(isolated_state["manifest"], scope)

    target = isolated_state["sandbox"] / path
    assert not target.exists()

    with pytest.raises(
        PermissionError,
        match="expected existing file",
    ):
        isolated_state["server"].write_file(path, "new contents")

    assert not target.exists()
    assert load_manifest(isolated_state["manifest"]) == scope


def test_overwrite_content_hash_mismatch_preserves_file_and_grant(
    isolated_state,
):
    path = "existing.txt"
    original = "original contents"
    approved_replacement = "approved replacement"
    attempted_replacement = "different replacement"

    target = isolated_state["sandbox"] / path
    target.write_text(original, encoding="utf-8")

    scope = overwrite_scope(path, original, approved_replacement)
    save_manifest(isolated_state["manifest"], scope)

    with pytest.raises(
        PermissionError,
        match="proposed content was not approved",
    ):
        isolated_state["server"].write_file(path, attempted_replacement)

    assert target.read_text(encoding="utf-8") == original
    assert load_manifest(isolated_state["manifest"]) == scope


def test_creation_grant_cannot_authorize_a_different_path(isolated_state):
    authorized_path = "authorized.txt"
    attempted_path = "other.txt"
    content = "approved content"

    scope = creation_scope(authorized_path, content)
    save_manifest(isolated_state["manifest"], scope)

    attempted_target = isolated_state["sandbox"] / attempted_path

    with pytest.raises(
        PermissionError,
        match="no valid one-time grant",
    ):
        isolated_state["server"].write_file(attempted_path, content)

    assert not attempted_target.exists()
    assert load_manifest(isolated_state["manifest"]) == scope


def test_empty_path_is_rejected(isolated_state):
    with pytest.raises(ValueError, match="Path cannot be empty"):
        isolated_state["server"].resolve_sandbox_path("")


def test_read_file_missing_target_raises_file_not_found(isolated_state):
    with pytest.raises(FileNotFoundError, match="File not found"):
        isolated_state["server"].read_file("missing.txt")


def test_list_files_returns_sorted_sandbox_relative_files(isolated_state):
    sandbox = isolated_state["sandbox"]
    (sandbox / "z.txt").write_text("z", encoding="utf-8")
    (sandbox / "nested").mkdir()
    (sandbox / "nested" / "a.txt").write_text("a", encoding="utf-8")

    result = isolated_state["server"].list_files()

    assert result == ["nested/a.txt", "z.txt"]

def test_read_file_returns_existing_file_contents(isolated_state):
    """A sandbox-relative read succeeds for an existing regular file."""
    server = isolated_state["server"]
    target = isolated_state["sandbox"] / "existing.txt"
    expected_content = "existing sandbox content\n"
    target.write_text(expected_content, encoding="utf-8")

    assert server.read_file("existing.txt") == expected_content


def test_overwrite_grant_cannot_authorize_a_different_path(isolated_state):
    """An overwrite grant is bound to its exact path."""
    server = isolated_state["server"]
    sandbox = isolated_state["sandbox"]
    manifest_path = isolated_state["manifest"]

    authorized_path = "authorized.txt"
    attempted_path = "other.txt"
    original_content = "original contents"
    proposed_content = "approved replacement"

    authorized_target = sandbox / authorized_path
    attempted_target = sandbox / attempted_path
    authorized_target.write_text(original_content, encoding="utf-8")

    scope = overwrite_scope(
        authorized_path,
        original_content,
        proposed_content,
    )
    save_manifest(manifest_path, scope)

    with pytest.raises(
        PermissionError,
        match="no valid one-time grant",
    ):
        server.write_file(attempted_path, proposed_content)

    assert authorized_target.read_text(encoding="utf-8") == original_content
    assert not attempted_target.exists()
    assert load_manifest(manifest_path) == scope

def test_read_file_rejects_directory(isolated_state):
    """Reading a directory through read_file must raise ValueError."""
    server = isolated_state["server"]
    directory = isolated_state["sandbox"] / "nested"
    directory.mkdir()

    with pytest.raises(ValueError, match="Path is not a file"):
        server.read_file("nested")

    assert directory.is_dir()
