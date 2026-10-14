
import hashlib
import json
from pathlib import Path

from fastmcp import FastMCP


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MCP_SANDBOX = (
    PROJECT_ROOT
    / "workspaces"
    / "mcp_sandbox"
).resolve()

# Host-managed control-plane state; this path is outside MCP_SANDBOX.
WRITE_SCOPE_FILE = (
    PROJECT_ROOT
    / ".copilot_runtime"
    / "active_write_scope.json"
)


def _load_write_scope() -> dict:
    """Load and strictly validate the content-bound authorization manifest."""
    try:
        scope = json.loads(
            WRITE_SCOPE_FILE.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise PermissionError(
            "Write denied: authorization manifest is missing or invalid."
        ) from exc

    if not isinstance(scope, dict):
        raise PermissionError(
            "Write denied: invalid authorization manifest."
        )

    required_keys = {
        "role",
        "task_id",
        "allowed_creations",
        "allowed_overwrites",
    }

    # Reject the legacy path-only schema instead of silently trusting it.
    if set(scope) != required_keys:
        raise PermissionError(
            "Write denied: manifest schema is invalid or outdated."
        )

    allowed_creations = scope["allowed_creations"]
    allowed_overwrites = scope["allowed_overwrites"]

    if not isinstance(allowed_creations, list):
        raise PermissionError(
            "Write denied: invalid creation grants."
        )

    if not isinstance(allowed_overwrites, list):
        raise PermissionError(
            "Write denied: invalid overwrite grants."
        )

    def validate_relative_path(value: object) -> bool:
        if (
            not isinstance(value, str)
            or not value
            or not value.strip()
            or value != value.strip()
            or "\\" in value
        ):
            return False

        candidate = Path(value)

        return (
            not candidate.is_absolute()
            and candidate.as_posix() == value
            and value not in {".", ".."}
            and ".." not in candidate.parts
            and "." not in candidate.parts
        )

    def validate_hash(value: object) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(ch in "0123456789abcdef" for ch in value)
        )

    creation_paths = []

    for rule in allowed_creations:
        if not isinstance(rule, dict):
            raise PermissionError(
                "Write denied: malformed creation grant."
            )

        if set(rule) != {"path", "approved_content_sha256"}:
            raise PermissionError(
                "Write denied: invalid creation grant fields."
            )

        if not validate_relative_path(rule.get("path")):
            raise PermissionError(
                "Write denied: invalid creation grant path."
            )

        if not validate_hash(rule.get("approved_content_sha256")):
            raise PermissionError(
                "Write denied: invalid creation content hash."
            )

        creation_paths.append(rule["path"])

    overwrite_paths = []
    required_overwrite_fields = {
        "path",
        "expected_existing_sha256",
        "approved_content_sha256",
    }

    for rule in allowed_overwrites:
        if not isinstance(rule, dict):
            raise PermissionError(
                "Write denied: malformed overwrite grant."
            )

        if set(rule) != required_overwrite_fields:
            raise PermissionError(
                "Write denied: invalid overwrite grant fields."
            )

        if not validate_relative_path(rule.get("path")):
            raise PermissionError(
                "Write denied: invalid overwrite grant path."
            )

        if not validate_hash(rule.get("expected_existing_sha256")):
            raise PermissionError(
                "Write denied: invalid existing content hash."
            )

        if not validate_hash(rule.get("approved_content_sha256")):
            raise PermissionError(
                "Write denied: invalid approved content hash."
            )

        overwrite_paths.append(rule["path"])

    # Avoid ambiguous duplicate grants and conflicting grant types.
    all_paths = creation_paths + overwrite_paths

    if len(all_paths) != len(set(all_paths)):
        raise PermissionError(
            "Write denied: duplicate or conflicting path grants."
        )

    if all_paths:
        _validate_task_identity(scope)
    elif scope["role"] is not None or scope["task_id"] is not None:
        _validate_task_identity(scope)

    return scope


def _save_write_scope(scope: dict) -> None:
    """Atomically persist a consumed authorization manifest."""
    temporary_path = WRITE_SCOPE_FILE.with_name(
        WRITE_SCOPE_FILE.name + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(scope, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(WRITE_SCOPE_FILE)


def _validate_task_identity(scope: dict) -> None:
    """Accept write grants only for an identified Implementer task."""
    role = scope.get("role")
    task_id = scope.get("task_id")

    if role != "implementer":
        raise PermissionError(
            "Write denied: only the implementer role may receive write grants."
        )

    if (
        not isinstance(task_id, str)
        or not task_id.strip()
        or len(task_id) > 128
        or any(ord(ch) < 32 for ch in task_id)
    ):
        raise PermissionError(
            "Write denied: authorization has no valid role/task identity."
        )


def authorize_and_consume_write(
    relative_path: str,
    content: str,
    target: Path,
) -> str:
    """
    Authorize one exact write and consume its single-use grant first.

    Creation grants bind the exact path and approved content hash.
    Overwrite grants additionally bind the current existing content hash.
    """
    scope = _load_write_scope()

    proposed_hash = hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()

    # Creation: exact path, target must not exist, exact content hash.
    for index, rule in enumerate(scope["allowed_creations"]):
        if rule["path"] != relative_path:
            continue

        if target.exists():
            raise PermissionError(
                f"Write denied: creation grant cannot overwrite "
                f"existing file '{relative_path}'."
            )

        _validate_task_identity(scope)

        if proposed_hash != rule["approved_content_sha256"]:
            raise PermissionError(
                f"Write denied: proposed content was not approved for "
                f"'{relative_path}'."
            )

        # Consume this grant before touching the target.
        scope["allowed_creations"] = [
            existing_rule
            for rule_index, existing_rule
            in enumerate(scope["allowed_creations"])
            if rule_index != index
        ]
        _save_write_scope(scope)
        return "create"

    # Overwrite: exact path, current hash, and approved new content hash.
    for index, rule in enumerate(scope["allowed_overwrites"]):
        if rule["path"] != relative_path:
            continue

        if not target.is_file():
            raise PermissionError(
                f"Write denied: expected existing file '{relative_path}'."
            )

        _validate_task_identity(scope)

        current_hash = hashlib.sha256(
            target.read_bytes()
        ).hexdigest()

        if current_hash != rule["expected_existing_sha256"]:
            raise PermissionError(
                f"Write denied: existing content hash changed for "
                f"'{relative_path}'."
            )

        if proposed_hash != rule["approved_content_sha256"]:
            raise PermissionError(
                f"Write denied: proposed content was not approved for "
                f"'{relative_path}'."
            )

        # Consume this grant before touching the target.
        scope["allowed_overwrites"] = [
            existing_rule
            for rule_index, existing_rule
            in enumerate(scope["allowed_overwrites"])
            if rule_index != index
        ]
        _save_write_scope(scope)
        return "overwrite"

    raise PermissionError(
        f"Write denied: no valid one-time grant for '{relative_path}'."
    )


mcp = FastMCP(
    "Local App Dev Copilot Tools"
)


def resolve_sandbox_path(
    relative_path: str,
) -> Path:

    if not relative_path.strip():
        raise ValueError(
            "Path cannot be empty."
        )

    candidate = (
        MCP_SANDBOX
        / relative_path
    ).resolve()

    try:
        candidate.relative_to(
            MCP_SANDBOX
        )
    except ValueError as exc:
        raise ValueError(
            "Path escapes the MCP sandbox."
        ) from exc

    return candidate


@mcp.tool
def list_files() -> list[str]:

    files = []

    for path in MCP_SANDBOX.rglob("*"):

        if path.is_file():

            files.append(
                str(
                    path.relative_to(
                        MCP_SANDBOX
                    )
                )
            )

    return sorted(files)


@mcp.tool
def read_file(
    path: str,
) -> str:

    target = resolve_sandbox_path(
        path
    )

    if not target.exists():

        raise FileNotFoundError(
            f"File not found: {path}"
        )

    if not target.is_file():

        raise ValueError(
            f"Path is not a file: {path}"
        )

    return target.read_text(
        encoding="utf-8"
    )


@mcp.tool
def write_file(
    path: str,
    content: str,
) -> str:

    target = resolve_sandbox_path(
        path
    )

    relative_path = target.relative_to(
        MCP_SANDBOX
    ).as_posix()

    authorize_and_consume_write(
        relative_path,
        content,
        target,
    )

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    target.write_text(
        content,
        encoding="utf-8",
    )

    return (
        f"Successfully wrote "
        f"{len(content)} characters to {path}"
    )


@mcp.tool
def delete_file(
    path: str,
) -> str:
    """Deletion is disabled because no role has deletion permission."""
    raise PermissionError("Deletion is disabled by policy.")


if __name__ == "__main__":
    mcp.run()
