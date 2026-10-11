# 查看与重建传感器质量报告

最终可视化入口：

```text
reproduction/isaac61/outputs/quality-iteration/final-quality-report-v2/index.html
```

页面展示原始参考图像、真实完整 HDF5 采集、四指接触标定，以及独立结构光几何控制。PNG 嵌入页面，MP4 和原始 HTML 查看器可以直接打开；状态、源文件 SHA256、源码与配置版本、实际后端和显示量程保留在报告中。各项测试结果与指标以 `report.json` 和输入的原始审计为准。

## 从现有数据重建

在 RoboDojo 根目录执行，使用已有完成目录：

```bash
.venv-isaac61/bin/python scripts/build_sensor_quality_report.py \
  --reference-dir reproduction/isaac61/outputs/quality-iteration/review-reference \
  --capture-dir reproduction/isaac61/outputs/quality-iteration/fullmodal-contact-final \
  --capture-viewer-dir reproduction/isaac61/outputs/quality-iteration/chunked-hdf5-viewer-final \
  --tactile-dir reproduction/isaac61/outputs/quality-iteration/tactile-contact-final \
  --sl-dir reproduction/isaac61/outputs/quality-iteration/sl-confidence-plane-final \
  --sl-dir reproduction/isaac61/outputs/quality-iteration/sl-confidence-tilted-final \
  --sl-dir reproduction/isaac61/outputs/quality-iteration/frozen-rgb-v2 \
  --output reproduction/isaac61/outputs/quality-iteration/final-quality-report-v2 \
  --copy-videos --require-passed
```

构建器只读取已有数据，不启动 Kit/GPU 模拟，也不生成或改变传感器图像。输出为 `index.html` 与 `report.json`。`--copy-videos` 将现有 MP4 复制到报告目录；原始 HDF5 保留在采集目录。离线暗色显示、筛选和完整 provenance 面板可直接使用。

最终完整数据保存于 `fullmodal-contact-final/episode.hdf5`；四指独立接触组件保存于 `tactile-contact-final/episode.hdf5`。两者使用不同 observation schema，组件测试不能替代 head、left_wrist、right_wrist 的全模态采集。旧 NPZ moving-wrist 记录保留为无接触基线，不会被重新标注为新的接触采集或后来源码版本。

## 报告验证什么

完整 HDF5 采集需要 `manifest.json` 的完整 HDF5 帧索引、明确通过且 `real_runtime: true` 的 `validation.json`，以及独立审计链接到真实 `source_hdf5` 和 `source_hdf5_sha256`。报告重新计算 HDF5 SHA256，逐帧读取并核对传感器形状、时钟、帧数、坐标与源 manifest、validation 的一致性。力字段包含真实姿态时，也验证原始官方 taxel 力到胶面坐标的转换。

组件文件明确使用 `robodojo.tactile_component.v1`，检查四指实际数据与时钟，不生成相机占位数据。结构光控制的 `report.json` 和独立审计保持各自状态，审计需通过 SHA256 链接原始报告和 NPZ 控制输入；这些控制不替代完整采集。控制目录的读入上限为 32 个 NPZ 文件、1 GiB。

缺少独立验收会显示 `incomplete`，失败验收保留为 `failed`。输入结构错误、帧索引冲突、无效时钟或源 SHA 不符会被拒绝。`--require-passed` 在证据未全部通过时返回退出码 1，报告仍保留真实状态，不会因为部分预览可打开而声称整体通过。

原生日志审计与实际进程退出证据分别保留；JSON 中的通过状态不能代替真实退出结果。参考视频仅用于定性比较，其提取记录保留视频和帧 SHA256，不作为新传感器验收或策略成功证据。全部新录制沿用自身真实源码与配置 provenance，旧数据不被赋予后来的 HEAD。

## 图像和力的含义

全模态 HTML 提供 RGB、光学深度差、官方 taxel Z/XY 力、胶面外法向 −Y 力和胶体 XZ 切向力。胶面压入的有符号法向力为负。原始官方 `normal_force`、`shear_force` 数组保持原样；胶面投影依赖保存的真实 taxel 和胶体 XYZW 姿态。力单位为 N，深度和压入量为 m。单指比例为 0–1，四指光学支持评分之和为 0–4。

PNG 与 HTML 起始帧选择实际记录中最强的胶面法向力帧，旧格式回退到原始 taxel Z 分量。四指组件预览的原始 `selected_frame` 是整数，选择准则为四指各自最大正向光学深度连通区域比例之和，取首次最大值；该和为 0–4，代表光学支持而非力。报告按原始阈值逐帧重算连通区域及封闭孔洞，检查所选真实物理帧、时间戳和阶段。生产者 JSON 保持原样，聚合报告另存验证后的选择结果。完整 HDF5 预览的选择记录则包含原始帧索引、评分与量程；报告重算其实际胶面力评分并验证整段最强帧。最终报告使用全部帧，不使用截断预览代替整段接触证据。

触觉彩色图像由官方 TacSL/Taxim 光学模型产生，没有后加标记、接触色块或逐帧自动增强。光学深度差代表可见接触几何变化，并非 FEM 胶体形变；力场是官方惩罚模型估计，逐 taxel 求和不是校准后的合力。标定螺母的加载、滑动、扭转和释放属于真实运动学夹具序列，不证明机器人策略抓取成功，也不等同于 UniVTAC UIPC 模型或真实硬件标定。

IR 使用原始 uint8 灰度与实际曝光，是 RTX 可见辐亮度和传感器响应代理，未模拟完整 NIR 材质光谱。GT 深度属于 RGB 光学坐标系；SL 深度属于左 IR 光学坐标系，来自双目匹配。SL 稀疏或无效像素保留 NaN，没有 GT 填充。深度、触觉热图使用固定跨帧物理量色标，颜色只影响显示，HDF5 原 dtype、NaN 和力符号保持不变。

不同参考场景、光照和接触模型的视觉差异不构成量化改进百分比。报告保留各原始结果及其限制，而不从外观推断未测量的精度或策略成功率。

新查看器按帧输出独立脚本，避免长录像超过 V8 单字符串上限。`--capture-viewer-dir` 验证单独的离线导出与原始 HDF5 路径、SHA256、全部帧和显示内容一致，再使用其图片、视频及 HTML 入口。原始采集、旧预览及审计不会被覆盖；报告分别保存原采集与离线查看器的来源。
