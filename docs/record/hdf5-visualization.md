# 同步 HDF5 采集与可视化

在 RoboDojo 根目录运行命令。最终完整记录位于：

```text
reproduction/isaac61/outputs/quality-iteration/fullmodal-contact-final/episode.hdf5
```

该文件保存 head、left_wrist、right_wrist 三视角 RGB、GT 深度、SL 深度、左右 IR；另保存左、右夹爪各两指的触觉 RGB、光学深度、力场及其坐标定义。原始数组、标定、机器人状态、动作、物理帧与模拟时间戳都来自同一次冻结物理状态的采集。

## 直接查看现有 HDF5

已验证的完整 100 帧逐帧入口为 `reproduction/isaac61/outputs/quality-iteration/chunked-hdf5-viewer-final/multimodal.html`。该离线导出来自上面的原始 HDF5，按帧使用独立脚本，避免长录像超过 V8 单字符串上限。原始采集及旧预览保留；总览入口为 `reproduction/isaac61/outputs/quality-iteration/final-quality-report-v2/index.html`。

```bash
bash reproduction/isaac61/run.sh visualize \
  reproduction/isaac61/outputs/quality-iteration/fullmodal-contact-final/episode.hdf5
```

输出 `multimodal.html`、`multimodal.mp4`、`multimodal.png` 和 `visualization.json`。HTML 可拖动帧、播放，并切换触觉显示。PNG 和 HTML 初始帧选择实际胶面外法向力绝对值最大的帧；旧记录没有胶面投影时使用官方 taxel Z 分量。没有力场时使用明确标注的光学变化评分。所选原始帧号、评分含义、源 HDF5 SHA256 与显示范围保存在 `visualization.json`。MP4 保持时间顺序，使用记录的模拟时间间隔计算播放频率。

查看胶面受力，或调整固定米制色标：

```bash
bash reproduction/isaac61/run.sh visualize \
  reproduction/isaac61/outputs/quality-iteration/fullmodal-contact-final/episode.hdf5 \
  --output reproduction/isaac61/outputs/quality-iteration/pad-force-preview \
  --tactile pad_normal --depth-min 0.1 --depth-max 2.5 \
  --delta-limit 0.0006 --force-limit 1.5 --shear-limit 1.5
```

| HTML/CLI 模式 | 显示内容 |
| --- | --- |
| `rgb` | 原始触觉 RGB 和米制光学深度 |
| `optical_depth_delta` | 无接触基线深度减去当前深度，保留正负值 |
| `normal_force` | 官方 taxel 局部 Z 分量，原始数组 |
| `shear_force` | 官方 taxel 局部 XY 切向力大小 |
| `pad_normal` | 实际胶体外法向 −Y 的有符号力，压入为负 |
| `pad_shear` | 实际胶体 XZ 切向力大小 |

所有图像保持原比例并留黑边。IR 直接显示原始 uint8 灰度，保留实际曝光。GT 与 SL 深度使用跨帧固定米制色标，分别属于 RGB 和左 IR 光学坐标系。SL 无有效匹配的像素保持 NaN，显示为紫色；不会用 GT、零值或插值结果填充。触觉热图按官方 x 外层、z 内层采样顺序显示为 z 行、x 列，HDF5 原数组顺序不变。颜色映射只用于显示。

导出长录像可使用 `--frame-stride`、`--max-frames` 限制 HTML 大小。导出器一次只读取一个原始帧，逐帧写视频和预览；独立 HTML 包含选定预览，浏览器会加载这些预览。最终质量报告使用全部帧，避免截断后把局部最强帧当作整段最强帧。

## 重新采集

使用新的空输出目录，避免覆盖已有结果：

```bash
bash reproduction/isaac61/run.sh collect --task stack_bowls \
  --env-cfg arx_x5_multimodal --frames 100 --contact-demo \
  --contact-phase-duration 0.4 --visualize \
  --output reproduction/isaac61/outputs/quality-iteration/my-contact-capture
```

接触演示驱动传感器附近真实的运动学标定螺母，包含基线、加载、保持、滑动、扭转、卸载和释放。它独立于任务对象与奖励布局，是标定序列，不代表机器人策略抓取成功。该模式使用 20×25 taxel 网格、胶体接触刚度 100；普通采集保留其配置的刚度，默认 1500。Isaac Sim 6.1 的运动学目标通过真实 CPU 原生缓冲区与 CUDA host readback 桥接，实际设置和后端写入 provenance。帧数不足以覆盖完整序列、采样间隔跨过阶段或控制器与传感器时钟不一致时明确失败。

真实低幅腕关节运动使用 `--motion-demo`，与 `--contact-demo` 互斥：

```bash
bash reproduction/isaac61/run.sh collect --task stack_bowls \
  --env-cfg arx_x5_multimodal --frames 12 --motion-demo --visualize \
  --output reproduction/isaac61/outputs/quality-iteration/my-moving-capture
```

## 传感器含义

触觉 RGB 来自官方 TacSL/Taxim 深度与光照模型。它保留模拟的真实接触轮廓与官方 RGB 输出，没有后加标记、接触色块或逐帧亮度增强。`optical_depth_delta` 是无接触基线相机深度减去当前 RTX 相机深度的有符号米制差，是可见几何的光学代理，并非 FEM 胶体形变或真实硬件标定结果。

原 `normal_force` 与 `shear_force` 是官方惩罚模型在 taxel 局部 Z、XY 轴上的估计，不能直接等同于胶面压力或力传感器读数。新增 `force_body_xyz`、`pad_normal_force`、`pad_shear_force_xz` 使用记录的真实 taxel 世界旋转和胶体世界旋转计算；相应 XYZW 姿态也保存在 HDF5，读回时独立重算验证。逐点力之和不作为校准后的合力。没有受支持的刚性接触查询时，相关力字段为 `None`，保留明确原因；光学图像仍可采集。

结构光 IR 是 RTX 可见辐亮度及传感器响应代理，包含独立双目环境背景与主动投影信号，不模拟完整 NIR 材质光谱。HDF5 另存投影差分 `ir_left_active`、`ir_right_active` 和原始 HDR 辐亮度，保留 dtype、NaN 与实际渲染阶段记录。SL 深度由双目匹配重建；GT 深度独立保存，不用于替代重建失败。

## 文件协议与读回

`HDF5Writer(directory, run_metadata)` 逐帧验证并写入 `/pending`，随后提交到 `/frames/00000000` 等节点，最后递增 `committed_frame_count`。`/time/physics_frame`、`/time/timestamp_ns` 是严格递增的 int64 索引。数值数组保存原 dtype 和 NaN；新增字段、内外参、机器人状态、字典元数据与绝对 USD 路径形式的键递归保存。

录制中为 `episode.hdf5.partial`。完整采集和场景、清理检查结束后，通过原子重命名发布 `episode.hdf5`。正常异常会记录失败原因并保留已提交帧；后续预览失败会撤回最终文件为失败 partial，删除通过标记。该协议不承诺硬断电损坏的 HDF5 仍可恢复。已有输出不会被覆盖。

```bash
bash reproduction/isaac61/run.sh python -c \
  'from utils.multimodal_hdf5 import validate_hdf5; print(validate_hdf5("reproduction/isaac61/outputs/quality-iteration/fullmodal-contact-final/episode.hdf5"))'
```

`HDF5Reader(path)` 提供 `len(reader)`、`sync(index)`、`read_frame(index)`、`metadata` 和上下文管理。验证逐帧检查 schema、类型、同步、传感器数据与力坐标变换，不把文件格式验证当作任务策略成功证明。`run.capture_provenance` 保存实际版本、源码提交、dirty 状态、源文件 SHA256 和查询到的渲染设置；原生日志及独立审计另行保留。

四指触觉独立标定文件位于 `quality-iteration/tactile-contact-final/episode.hdf5`，schema 为 `robodojo.tactile_component.v1`，不包含三视角全模态数据。使用专用查看器：

```bash
bash reproduction/isaac61/run.sh python reproduction/isaac61/visualize_tactile_component.py \
  reproduction/isaac61/outputs/quality-iteration/tactile-contact-final/episode.hdf5
```

`--also-npz` 可另存旧 NPZ 帧，最终主文件仍为 HDF5。已有 NPZ 录像可以逐帧转换，保留原始 manifest、验证、日志审计及文件 SHA256，并明确标注为格式转换：

```bash
bash reproduction/isaac61/run.sh python reproduction/convert_multimodal_hdf5.py \
  reproduction/isaac61/outputs/existing-recording --output reproduction/isaac61/outputs/converted-recording
```
