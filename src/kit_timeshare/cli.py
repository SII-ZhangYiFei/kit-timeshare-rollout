"""Read-only environment checks and isolated launch commands."""

import argparse, json, shlex
from dataclasses import asdict
from .compat import matrix, version


def main():
    p = argparse.ArgumentParser(
        description="Independent CPU PhysX rollout with a shared Kit renderer"
    )
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "doctor", help="Inspect versions and physical GPU mapping; does not launch Kit"
    )
    d = sub.add_parser(
        "docker-command", help="Print an isolated Docker command; does not run it"
    )
    d.add_argument("--gpu-uuid", required=True)
    d.add_argument("--image", required=True)
    d.add_argument("args", nargs=argparse.REMAINDER)
    a = p.parse_args()
    if a.command == "doctor":
        from .gpu import discover, accessible_nodes

        result = {
            "expected": matrix(),
            "installed": {
                k: version(k) for k in ["isaacsim", "omnigibson", "isaaclab"]
            },
        }
        try:
            result.update(
                gpus=[asdict(g) for g in discover()], device_nodes=accessible_nodes()
            )
        except RuntimeError as exc:
            result["gpu_error"] = str(exc)
        print(json.dumps(result, indent=2))
    else:
        from .gpu import docker_command

        args = a.args[1:] if a.args[:1] == ["--"] else a.args
        print(shlex.join(docker_command(a.gpu_uuid, a.image, args or ["bash"])))


if __name__ == "__main__":
    main()
