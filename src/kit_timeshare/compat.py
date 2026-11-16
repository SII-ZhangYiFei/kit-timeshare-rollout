from __future__ import annotations
from importlib import metadata, resources, util
from pathlib import Path
import json, subprocess


def matrix():
    return json.loads(
        resources.files("kit_timeshare").joinpath("compatibility.json").read_text()
    )


def version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def check_versions(backend, root=None, strict=True):
    expected = matrix()
    actual = {
        "isaacsim": version("isaacsim"),
        expected[backend]["distribution"]: version(expected[backend]["distribution"]),
    }
    problems = []
    if actual["isaacsim"] != expected["isaacsim"]:
        problems.append(
            f"isaacsim: expected {expected['isaacsim']}, found {actual['isaacsim']}"
        )
    name = expected[backend]["distribution"]
    if actual[name] != expected[backend]["version"]:
        problems.append(
            f"{name}: expected {expected[backend]['version']}, found {actual[name]}"
        )
    if root is None and backend == "behavior":
        spec = util.find_spec("omnigibson")
        if spec and spec.origin:
            root = Path(spec.origin).parent
    commit = None
    if root:
        p = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        if p.returncode == 0:
            commit = p.stdout.strip()
            if commit != expected[backend]["git_commit"]:
                problems.append(
                    "Upstream checkout commit differs from the validated compatibility profile"
                )
    if strict and problems:
        raise RuntimeError(
            "Unsupported SDK profile: "
            + "; ".join(problems)
            + ". See compatibility.json; unverified versions require strict_versions=False explicitly."
        )
    return {
        "packages": actual,
        "upstream_commit": commit,
        "problems": problems,
        "strict": strict,
    }
