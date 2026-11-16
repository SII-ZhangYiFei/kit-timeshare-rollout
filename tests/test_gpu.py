import stat, types, unittest
from unittest.mock import patch
from kit_timeshare import gpu


class Node:
    def __init__(self, name):
        self.name = name

    def is_char_device(self):
        return True

    def __str__(self):
        return "/dev/" + self.name


class GPUTests(unittest.TestCase):
    def test_denied_ghost_node_does_not_count_as_exposed_gpu(self):
        nodes = [Node("nvidia0"), Node("nvidia1")]

        def opened(node, flags):
            if node.name == "nvidia0":
                raise PermissionError("device cgroup denies GPU1")
            return 12

        with patch.object(gpu.Path, "glob", return_value=nodes), patch.object(
            gpu.os, "open", side_effect=opened
        ), patch.object(gpu.os, "close") as closed:
            self.assertEqual(gpu.accessible_nodes(), ["/dev/nvidia1"])
            closed.assert_called_once_with(12)

    def test_docker_command_binds_physical_uuid_and_render_node(self):
        g = gpu.GPU("GPU-test", "0000:01:00.0", 1, "/dev/dri/renderD129", "RTX", "580")
        with patch.object(gpu, "discover", return_value=[g]):
            command = gpu.docker_command(g.uuid, "my-sdk", ["python", "demo.py"])
        self.assertIn("device=GPU-test", command)
        self.assertIn("/dev/dri/renderD129", command)
        self.assertEqual(command[-3:], ["my-sdk", "python", "demo.py"])

    def test_multi_gpu_nodes_rejected_before_sdk(self):
        g = gpu.GPU("GPU-test", "0000:01:00.0", 1, "/dev/dri/renderD129", "RTX", "580")
        with patch.object(gpu, "discover", return_value=[g]), patch.object(
            gpu, "accessible_nodes", return_value=["/dev/nvidia0", "/dev/nvidia1"]
        ), patch.object(gpu.Path, "glob", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "Multiple GPU"):
                gpu.select_isolated_gpu(g.uuid)


if __name__ == "__main__":
    unittest.main()
