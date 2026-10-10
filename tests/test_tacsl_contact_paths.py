import pytest

from env.tactile_manager.contact_paths import contact_path_for_env


@pytest.mark.parametrize(
    "template",
    [
        "{ENV_REGEX_NS}/bowl",
        "/World/envs/env_[^/]+/bowl",
        "/World/envs/env_.*/bowl",
        "/World/envs/env_.+/bowl",
        "/World/envs/env_0/bowl",
    ],
)
def test_each_environment_contact_is_concrete_and_isolated(template):
    assert contact_path_for_env(template, 3) == "/World/envs/env_3/bowl"
    assert contact_path_for_env(template, 0) == "/World/envs/env_0/bowl"


def test_deliberately_global_rigid_body_is_preserved():
    assert contact_path_for_env("/World/Probe_env0/mesh", 3) == "/World/Probe_env0/mesh"
    assert contact_path_for_env("/World/probe/env_0/mesh", 3) == "/World/probe/env_0/mesh"


@pytest.mark.parametrize("path", ["relative", "/World/obj.*", "{OTHER}/mesh", "/World/envs/env_2/bowl"])
def test_bad_or_cross_environment_contact_paths_fail(path):
    with pytest.raises(ValueError):
        contact_path_for_env(path, 3)
