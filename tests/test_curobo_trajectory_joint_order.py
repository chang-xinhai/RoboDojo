"""CPU regression for named cuRobo trajectory columns without importing CUDA."""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest


class CpuTensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def detach(self):
        return self

    def cpu(self):
        return self.values


@pytest.mark.parametrize('batched', [False, True])
def test_named_equal_dof_trajectory_preserves_active_joint_order(batched):
    path = Path(__file__).parents[1] / 'env/planner_manager/curobo_planner.py'
    tree = ast.parse(path.read_text())
    method = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == '_extract_trajectory')
    scope = {'np': np}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), scope)
    planner = SimpleNamespace(active_joints_name=['a', 'b'],
                              cspace_joint_names=['a', 'b'], _active_to_cspace_idx=[0, 1])
    extract = scope['_extract_trajectory']
    values = np.array([[2., 1.], [4., 3.]])
    if batched:
        values = values[None]
    actual = extract(planner, CpuTensor(values), batched, ['b', 'a'])
    np.testing.assert_array_equal(actual, values[..., [1, 0]])
    np.testing.assert_array_equal(extract(planner, CpuTensor(values), batched), values)
    with pytest.raises(ValueError, match='do not contain all active joints'):
        extract(planner, CpuTensor(values), batched, ['a', 'missing'])
