"""Static room geometry; its saved local transform retains the env ancestor."""
from .geometry import GeometryObject


class Room(GeometryObject):
    def __init__(self, prim_path, usd_path=None, instance_config=None):
        cfg = instance_config or {}
        super().__init__(prim_path, usd_path,
                         {'physics': {'collision': cfg.get('collision', True)}}, [0, 0, 0],
                         cfg.get('default_pos', [0, 0, 0]), cfg.get('default_ori', [0, 0, 0, 1]),
                         cfg.get('scale', [1, 1, 1]))
