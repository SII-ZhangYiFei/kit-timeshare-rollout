"""Execute source transformation against realistic fixtures without importing SDKs."""

import ast, hashlib, inspect, json, tempfile, textwrap, types, unittest
from pathlib import Path
import kit_timeshare

SOURCE = Path(kit_timeshare.__file__).parent / "backends/robodojo.py"


def functions():
    tree = ast.parse(SOURCE.read_text())
    tree.body = [
        x
        for x in tree.body
        if isinstance(x, ast.FunctionDef)
        and x.name in ["prepare_experience", "patch_reward_manager"]
    ]
    scope = dict(
        ast=ast,
        hashlib=hashlib,
        inspect=inspect,
        json=json,
        textwrap=textwrap,
        types=types,
        Path=Path,
    )
    exec(compile(tree, str(SOURCE), "exec"), scope)
    return scope


class RewardFixture:
    num_envs = 2

    def __init__(self):
        self.calls = [0, 0]
        self.final = [0, 0]
        self.score = [0, 0]

    def get_reward(self):
        reward_lst = []
        for env_idx in range(self.num_envs):
            self.calls[env_idx] += 1
            reward_lst.append(self.calls[env_idx] / 10)
        self._final_check()
        self._final_score()
        return reward_lst

    def _final_check(self):
        for env_idx in range(self.num_envs):
            self.final[env_idx] += 1

    def _final_score(self):
        for env_idx in range(self.num_envs):
            self.score[env_idx] += 1


class HooksTests(unittest.TestCase):
    def test_reward_preserves_inactive_rows_and_output_shape(self):
        rm = RewardFixture()
        router = types.SimpleNamespace(active=1)
        with tempfile.TemporaryDirectory() as d:
            functions()["patch_reward_manager"](rm, router, Path(d))
            self.assertEqual(rm.get_reward(), [0, 0.1])
            self.assertEqual(rm.calls, [0, 1])
            self.assertEqual(rm.final, [0, 1])
            self.assertEqual(rm.score, [0, 1])
            rm._timeshare_rewards[1] = 0.1
            router.active = 0
            self.assertEqual(rm.get_reward(), [0.1, 0.1])
            self.assertEqual(rm.calls, [1, 1])

    def test_private_experience_does_not_edit_sdk(self):
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            src = repo / "app.kit"
            original = '[dependencies]\n"isaacsim.asset.importer.urdf" = {version = "2.4.31", exact=true}\npath="${app}/../source"\n'
            src.write_text(original)
            options = types.SimpleNamespace(
                experience=src, urdf_importer_version="2.4.30"
            )
            out = functions()["prepare_experience"](repo, repo, options)
            self.assertEqual(src.read_text(), original)
            self.assertIn("2.4.30", out.read_text())
            self.assertNotIn("${app}", out.read_text())

    def test_unknown_template_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "app.kit"
            src.write_text("[dependencies]\n")
            with self.assertRaises(RuntimeError):
                functions()["prepare_experience"](
                    Path(d),
                    Path(d),
                    types.SimpleNamespace(
                        experience=src, urdf_importer_version="2.4.30"
                    ),
                )


if __name__ == "__main__":
    unittest.main()


class RegisteredRewardTests(unittest.TestCase):
    def test_registered_predicate_owner_is_retained_when_snapshot_copied(self):
        import copy

        tree = ast.parse(SOURCE.read_text())
        tree.body = [
            x
            for x in tree.body
            if isinstance(x, ast.FunctionDef) and x.name == "copy_state"
        ]
        scope = {"copy": copy}
        exec(compile(tree, str(SOURCE), "exec"), scope)

        class Owner:
            def __deepcopy__(self, memo):
                raise AssertionError("Must not clone native SDK owner")

            def predicate(self):
                return True

        owner = Owner()
        original = [("condition", {"callback": owner.predicate, "args": [1, 2]})]
        copied = scope["copy_state"](original)
        self.assertIs(copied[0][1]["callback"].__self__, owner)
        copied[0][1]["args"].append(3)
        self.assertEqual(original[0][1]["args"], [1, 2])


class RewardAliasingTests(unittest.TestCase):
    def test_registered_argument_mutation_cannot_leak_to_other_environment(self):
        import copy

        tree = ast.parse(SOURCE.read_text())
        tree.body = [
            x
            for x in tree.body
            if isinstance(x, ast.FunctionDef)
            and x.name in ["copy_state", "isolate_reward_rows"]
        ]
        scope = {"copy": copy}
        exec(compile(tree, str(SOURCE), "exec"), scope)
        args = {"label": "bowl0"}
        checks = [("is_axis_up", args)]
        parser = types.SimpleNamespace(pre_state=[{}, {}])
        manager = types.SimpleNamespace(
            num_envs=2, check_list=[[checks], [checks]], func_parser=parser
        )
        scope["isolate_reward_rows"](manager)
        manager.check_list[0][0][0][1]["env_idx"] = 0
        self.assertNotIn("env_idx", manager.check_list[1][0][0][1])
        self.assertNotIn("env_idx", args)
