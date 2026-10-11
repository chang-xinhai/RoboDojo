# 同步 HDF5 采集与可视化

在 RoboDojo 根目录执行：

```bash
bash reproduction/isaac61/run.sh collect --task stack_bowls \
  --env-cfg arx_x5_multimodal --frames 12 --motion-demo --visualize \
  --output reproduction/isaac61/outputs/hdf5-moving
bash reproduction/isaac61/run.sh visualize \
  reproduction/isaac61/outputs/hdf5-moving/episode.hdf5 \
  --depth-min 0.1 --depth-max 2.5 --frame-stride 1
```

最终数据为 `episode.hdf5`。生成的 `multimodal.html` 可逐帧播放、选择触觉 RGB、光学深度差、法向力或切向力；`multimodal.mp4` 按模拟时间间隔播放；`multimodal.png` 选择实际法向力绝对值最大的帧。图像保持比例并留黑边。`visualization.json` 记录所选帧、评分含义、源 HDF5 SHA256 和固定显示范围。如果没有力场，选择依据是光学变化，并明确注明无法证明受力。

显示不会修改原始数据：IR 使用原始 uint8 亮度；GT 和 SL 使用固定的米制色标，分别属于 RGB 和左 IR 光学坐标系，紫色表示无效深度。触觉 `optical_depth_delta` 是无接触基线光学深度减去当前 RTX 光学深度的有符号米制差；保留负值，是光学几何代理，并非 FEM 胶体形变。法向和切向数据是官方 TacSL 惩罚模型估计，逐采样点和不等于校准后的合力。力场不可用时保留 `None` 及原因。

`--tactile pad_normal` 显示胶面外法向 −Y 的有符号力估计（压入为负），`--tactile pad_shear` 显示胶体 XZ 切向力大小；HTML 也可切换。原 `normal_force` 是官方 taxel 局部 Z 分量，原 `shear_force` 是官方局部 XY 分量，二者保持原样，不能直接等同于胶面受压力。新增 `force_body_xyz`、`pad_normal_force`、`pad_shear_force_xz` 由记录的真实 taxel 世界旋转和胶体世界旋转转换得到，原始 XYZW 姿态一并保存并独立重算验证。采样热图遵循官方 x 外层、z 内层排列，显示为 z 行、x 列，不改原数组。

使用 `--delta-limit`、`--force-limit`、`--shear-limit` 设置整段视频一致的显示量程。使用 `--max-frames` 和 `--frame-stride` 控制 HTML 大小。导出时只读一个原始帧并逐个写入预览；独立 HTML 包含所选预览，浏览器会加载这些预览，较长录像应限制导出帧数。HDF5 本身保持全部帧。

## 文件协议与失败行为

`utils.multimodal_hdf5.HDF5Writer(directory, run_metadata)` 的 `append(observation)` 验证三视角 RGB、GT 深度、SL 深度、双目 IR，以及左右各两指 TacSL 数据。帧首先写入 `/pending`，随后提交到 `/frames/00000000` 等节点，最后递增 `committed_frame_count`。`/time/physics_frame` 和 `/time/timestamp_ns` 是严格递增的 int64 时间索引。数组保持 dtype 与 NaN；新增投影响应原图、HDR 辐亮度、内外参、机器人状态、动作和字典元数据递归无损保存。绝对 USD 路径形式的字典键保持原义。

录制中为 `episode.hdf5.partial`；成功提交全部数据、完成场景和环境清理验证后，以原子重命名发布 `episode.hdf5`。读写器仅暴露已提交帧，正常异常会记录 `status=failed`、失败原因并保留已提交数据。预览失败会撤回最终文件为失败 partial，同时删除通过标记。该协议不承诺硬断电后损坏的 HDF5 文件仍可恢复。`seal()` 关闭写入，`complete(validation=...)` 发布，`fail(error)` 标记失败。已有文件不会被覆盖。

`HDF5Reader(path)` 提供 `len(reader)`、`sync(index)`、`read_frame(index)`、`metadata` 和上下文管理；`validate_hdf5(path)` 逐帧验证 schema、类型、同步、可用传感器和时间索引，不把文件格式验证当作任务策略成功证明。`run.capture_provenance` 保存实际版本、源码提交、dirty 状态和查询到的渲染设置；原始每帧传感器 provenance 保留实际后端和坐标定义。外部原生日志审计仍单独进行。

`observation_schema='robodojo.tactile_component.v1'` 明确表示独立四指触觉组件测试，要求真实四指和时钟，不允许充当三相机全模态验收；完整 viewer 拒绝该 schema。

## 兼容与接触序列

`--also-npz` 可另存旧 NPZ 帧，canonical 文件仍为 HDF5。转换已有 NPZ 录像：

```bash
bash reproduction/isaac61/run.sh python reproduction/convert_multimodal_hdf5.py \
  reproduction/isaac61/outputs/existing-recording
```

转换逐帧读取，保留原始 manifest、验证、日志审计、文件 SHA256；标注为格式转换，不声称重新运行模拟器。也可通过 `--output` 写入空目录。

`--contact-demo --frames 100 --contact-phase-duration 0.4` 驱动挂载传感器附近真实的运动学标定螺母，记录基线、加载、滑动、扭转和卸载。它是独立标定夹具，不能声称机器人策略抓取成功；与 `--motion-demo` 互斥。末尾验证要求实际测量到全部阶段，帧数不足会明确失败。源码中的控制与触觉响应验证仍需真实 GPU 运行完成后才能用于验收。
