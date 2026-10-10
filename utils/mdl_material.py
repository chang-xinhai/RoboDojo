"""Native MDL material creation for Isaac Sim 6.1."""


def create_mdl_material(mtl_url, mtl_name, mtl_path):
    """Author a real MDL shader using the supported Omni USD command.

    Isaac Sim 6.1 removed the replicator behavior utility. CreateMdlMaterialPrim
    remains the API used by official experimental OmniPBR/OmniGlass wrappers.
    Preserve the old utility's texture inputs without substituting a material.
    """
    import omni.kit.commands
    import omni.usd
    from pxr import Sdf, UsdShade

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("MDL material creation requires an active USD stage")
    if stage.GetPrimAtPath(mtl_path):
        raise ValueError(f"MDL material target already exists: {mtl_path}")
    success, _ = omni.kit.commands.execute(
        "CreateMdlMaterialPrim", mtl_url=mtl_url, mtl_name=mtl_name, mtl_path=mtl_path, stage=stage
    )
    material = UsdShade.Material.Get(stage, mtl_path)
    shader = UsdShade.Shader.Get(stage, str(mtl_path) + "/Shader")
    if not success or not material or not shader:
        raise RuntimeError(f"Native MDL material creation failed: {mtl_url}:{mtl_name} at {mtl_path}")
    for name, value_type in (
        ("diffuse_texture", Sdf.ValueTypeNames.Asset),
        ("project_uvw", Sdf.ValueTypeNames.Bool),
        ("texture_scale", Sdf.ValueTypeNames.Float2),
        ("texture_rotate", Sdf.ValueTypeNames.Float),
    ):
        shader.CreateInput(name, value_type)
    return material
