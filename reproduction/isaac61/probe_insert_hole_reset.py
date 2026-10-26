import argparse,json,os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ["ROBODOJO_ASSET_ROOT"]=os.environ.get("ROBODOJO_INSERT_HOLE_ASSET_ROOT","/mnt/robo/yutong/tmp/robodojo-insert-hole-assets")
from reproduction.isaac61.prepare_insert_hole_assets import stage_assets
stage_assets()
from isaaclab.app import AppLauncher
from utils.isaaclab_compat import add_app_launcher_args,enable_legacy_isaacsim_extensions
from utils.load_file import load_yaml
from utils.task_layout import load_task_layout
from utils.task_runtime_config import configure_task_runtime,reset_task_episode
from omegaconf import OmegaConf
parser=argparse.ArgumentParser();add_app_launcher_args(parser,AppLauncher);parser.add_argument("--output",required=True);a=parser.parse_args();a.enable_cameras=False
launcher=AppLauncher(a);app=launcher.app
out={}
try:
 enable_legacy_isaacsim_extensions()
 from task.RoboDojo.task_registry import load_task_class
 eval_cfg=load_yaml(str(ROOT/"env_cfg/arx_x5.yml")); c=eval_cfg["config"]
 cfg=OmegaConf.create({k:load_yaml(str(ROOT/"env_cfg"/k/f"{c[k]}.yml")) for k in ("sim","scene","camera","robot")})
 cfg.no_render=True;cfg.task_env=load_yaml(str(ROOT/"task/RoboDojo/config/migration_insert_hole.yml"))
 layout,prov=load_task_layout(ROOT,"migration_insert_hole",layout_file=ROOT/"reproduction/isaac61/layouts/insert_hole.json");cfg.saved_layouts=[layout]
 profile=configure_task_runtime(cfg,"migration_insert_hole","arx_x5",eval_cfg,prov,enable_planner=False)
 _,cls=load_task_class("migration_insert_hole");env=cls(cfg,app,seed=0);reset_task_episode(env,profile)
 name=env.scene_manager.layout_manager.get_instance_name(0,label="prism");obj=env.scene_manager.layout_manager.get_scene_object(0,name)
 obj.apply_saved_pose()
 rows=[]
 for i in range(201):
  if i in (0,1,2,5,10,20,50,100,150,200):
   p,q=obj.get_world_pose()
   if hasattr(p,"detach"):p=p.detach().cpu().numpy()
   if hasattr(q,"detach"):q=q.detach().cpu().numpy()
   lp,lq=env.scene_manager.layout_manager.get_instance_pose(0,label="prism",relative=False)
   rows.append({"step":i,"native_p":np.asarray(p).reshape(-1).tolist(),"native_q":np.asarray(q).reshape(-1).tolist(),"layout_p":np.asarray(lp).reshape(-1).tolist(),"layout_q":np.asarray(lq).reshape(-1).tolist()})
  env.sim_step(render=False)
 out={"status":"ok","rows":rows}
except BaseException as e:
 out={"status":"error","error":f"{type(e).__name__}: {e}"}
finally:
 Path(a.output).write_text(json.dumps(out,indent=2))
 try: app.close(wait_for_replicator=False,skip_cleanup=True,exit_code=0)
 except Exception: pass
