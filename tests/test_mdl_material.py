"""USD authoring contracts; real MDL rendering is checked by live task smoke."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from utils.mdl_material import create_mdl_material


def bind_command(monkeypatch, stage, execute):
    omni = ModuleType("omni")
    kit = ModuleType("omni.kit")
    commands = ModuleType("omni.kit.commands")
    usd = ModuleType("omni.usd")
    commands.execute = execute
    usd.get_context = lambda: SimpleNamespace(get_stage=lambda: stage)
    omni.kit, omni.usd, kit.commands = kit, usd, commands
    for name, module in (("omni", omni), ("omni.kit", kit), ("omni.kit.commands", commands), ("omni.usd", usd)):
        monkeypatch.setitem(sys.modules, name, module)


def test_mdl_source_asset_and_identifier_are_preserved(monkeypatch):
    Usd = pytest.importorskip("pxr.Usd")
    UsdShade = pytest.importorskip("pxr.UsdShade")
    Sdf = pytest.importorskip("pxr.Sdf")
    stage = Usd.Stage.CreateInMemory()

    def execute(command, **kwargs):
        assert command == "CreateMdlMaterialPrim"
        assert kwargs["stage"] == stage
        material = UsdShade.Material.Define(stage, kwargs["mtl_path"])
        shader = UsdShade.Shader.Define(stage, kwargs["mtl_path"] + "/Shader")
        shader.SetSourceAsset(Sdf.AssetPath(kwargs["mtl_url"]), "mdl")
        shader.SetSourceAssetSubIdentifier(kwargs["mtl_name"], "mdl")
        material.CreateSurfaceOutput("mdl").ConnectToSource(shader.CreateOutput("out", Sdf.ValueTypeNames.Token))
        return True, None

    bind_command(monkeypatch, stage, execute)
    material = create_mdl_material("/fixtures/Linen_Blue.mdl", "Linen_Blue", "/World/Looks/water")
    shader = UsdShade.Shader.Get(stage, str(material.GetPath()) + "/Shader")
    assert shader.GetSourceAsset("mdl").path == "/fixtures/Linen_Blue.mdl"
    assert shader.GetSourceAssetSubIdentifier("mdl") == "Linen_Blue"
    assert shader.GetInput("texture_scale").GetTypeName() == Sdf.ValueTypeNames.Float2


def test_native_mdl_failure_raises_instead_of_returning_a_substitute(monkeypatch):
    Usd = pytest.importorskip("pxr.Usd")
    stage = Usd.Stage.CreateInMemory()
    bind_command(monkeypatch, stage, lambda *_args, **_kwargs: (False, None))
    with pytest.raises(RuntimeError, match="Native MDL material creation failed"):
        create_mdl_material("/fixtures/Linen_Blue.mdl", "Linen_Blue", "/World/Looks/water")
    assert not stage.GetPrimAtPath("/World/Looks/water")
