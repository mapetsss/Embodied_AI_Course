# a001 工程交付：A 双视图前端原型

2026-11-04。本机完成开发与 CPU 合成检查，目标机尚未部署或运行。没有真实模型、GPU/扩展、完整 SLAM 或 ATE 结果，不能据此声称精度提升或实时性。

## 基底与修改范围

根提交 `e6f4e3d474fad0e11f561482012be864ba8c3f17`；目标 lietorch 提交 `e7df86554156b36846008d8ddbcc4d8521a16554`。三份指南兼容补丁已在本机逐份应用并以 `git apply --reverse --check` 复核；具体指纹、受保护文件哈希和递归子模块提交见 `manifest.json`。A 的 `a.patch` 由 `guide-base/files/` 保存的**兼容基底原文**生成，未夹带 CUDA/模型加载兼容补丁。

只修改两个原文件：

| 文件 | 必要改动 |
|---|---|
| `mast3r_slam/mast3r_utils.py` | `mast3r_match_asymmetric(..., return_descriptors=False)` 可选返回第九项 `(Dff, Dkf)`。默认仍是八项；两幅 HWC 描述子来自同一次原有 decoder 调用，没有追加推理 |
| `mast3r_slam/tracker.py` | 按模式调用 A；已标定求解器增加可选权重、迭代上限与状态返回记录；沿用同一个原求解循环，没有另复制 tracker |

新增 `mast3r_slam/reliability.py`，以及 `config/a_off.yaml`、`a_diagnostic.yaml`、`a_weighted.yaml`。没有修改 frame、geometry、global_opt、main、dataloader、CUDA/C++、二进制接口或共享数据结构。后端继续原 C/Q 与优化逻辑；融合仍由原 `update_pointmap` 根据采纳后的位姿进行，未实现地图保护 P。B、长期记忆和异步更新未实现。

## 源码事实与实际公式

- `mast3r_asymmetric_inference` 直接调用 decoder 的两侧 head；DUSt3R `forward` 中把第二幅输出命名为 `pts3d_in_other_view`，明确其点图也在第一幅参考相机系。因此当前对的 `Xff`、`Xkf` 都在当前帧相机系，不能把 `Xkf.z` 当关键帧深度。
- 当前帧 `frame.update_pointmap(Xff,Cff)` 得到当前相机系 canonical 点图；关键帧 canonical 点图由原 `T_CkCf.act(Xkf)` 对齐后融合，在关键帧坐标系内。
- `matching.prep_for_iter_proj` 把 `X21` 的每个关键帧像素投到 `X11`，`pixel_to_lin` 使用 `u+w*v`。故虽然变量名是 `idx_f2k`，**向量按关键帧像素排列，每项指向当前帧像素**。`Xf` 已经由原 `get_points_poses` 按该索引 gather，源描述子也必须同样 gather。
- MASt3R `catmlp_dpt_head.reg_desc` 对最后一维做 L2 归一化；匹配内核用跨两侧描述子的点积。首版只比较同一 decoder 对的两幅描述子，插值后再归一化，不假设跨不同图像对/缓存调用可比。
- 原 `Intrinsics.K_frame` 已包含图像缩放与中心裁剪，主程序把它放在 keyframes.K；已标定原求解先把 canonical 点投回该 K 的射线。A 消费同一个 K，使用相同 canonical 点和 Sim(3) 尺度，不二次缩放 K。仅支持 `img_downsample=1`；原分支在降采样时没有同步缩放所有投影尺寸，首版直接拒绝启用 A，避免扩大修复范围。也拒绝无标定模式。关闭 A 时这两类原分支照常可用。

对第 k 个关键帧像素，令 `i = idx_f2k[k]`：

```text
T_CkCf = T_WCk.inv() * T_WCf                  # 原前端初估
X_static[k] = T_CkCf.act(Xf[k])              # Xf[k] 已由 i gather
uv_static[k] = project(K_key, X_static[k])
e[k] = clamp(1 - cosine(Dff_flat[i], sample(Dkf, uv_static[k])), 0, 2)
w[k] = clamp(exp(-e[k]/temperature), min_weight, 1)  # 仅确认可见时
```

源帧 `f` 到关键帧 `k` 的静态投影，不使用实际匹配的目标像素作为特征比较位置。只存在当前帧和最近关键帧这一对视图；没有多视图聚合或持久描述子缓存。

`grid_sample` 使用 `align_corners=False`，整数像素为中心，归一化为 `2*(u+0.5)/W-1` 与 `2*(v+0.5)/H-1`；HWC 与行优先展平一致。特征与点图必须同分辨率；不接受隐式特征 resize。边界是 `[0,W-1]×[0,H-1]`，微小浮点投影越界也保守地按未知处理。

## 可见性与中性规则

状态码见 `reliability.STATUS`，诊断 NPZ 保存 `status`。优先级为无效 → 非正深度 → 越界 → 缺少深度 → 遮挡 → 深度不一致 → 无效特征 → 可见。

关键帧 canonical 深度和置信度在静态投影处双线性采样。插值邻域必须有效且置信度超过原 C 阈值；特征邻域也必须有限、非零。容差是 `depth_abs_tol + depth_rel_tol*abs(depth)`，默认 `0.05 + 0.10*abs(depth)`，单位是该 canonical 点图的尺度，不能解释为真实深度传感器的米精度保证。

预测点明显在目标深度后面为遮挡；明显在前面为几何不一致未知。无效、非正深度、越界、遮挡、缺失和未知一律保持 `w=1`，不当作动态。只有可见且深度一致时才计算低权重。该分数是静态一致性启发式，不是校准后的运动概率。运行时没有读取真实轨迹或真实深度；GT 仅由事后的 evo 使用。

## 重估与回退

`off` 或没有 `a` 字段：不请求描述子扩展，不导入可靠性模块、不计算权重、不写诊断，沿用原推理、匹配、阈值、求解、融合和关键帧决策。只打印一次 `A_CONFIG mode=off`。

`diagnostic`：先完成原跟踪，再计算和保存诊断，返回原初估对象，重估次数为零。

`weighted`：原初估 → 一次可靠性计算并固定 → 同一个已标定 Sim(3) 求解器重估 → 检查后采纳。默认最多 5 步，配置硬上限 10 步，不重新计算权重或无限交替。原 `valid_match`、C/Q 阈值和 `valid_opt` 保持原语义，没有先把 Q 替换成 Q*w。

原 tracker 的 Huber 是**逐残差分量**，首版保持它：

```text
E(T) = sum_observation w[p] * sum_component rho_huber((S0*r(T))[p,c])
S0 = original valid masks * sqrt(Q) / sigma_pixel_or_log_depth
Huber weight = huber(S0*r, original huber threshold)
S = S0 * sqrt(Huber weight) * sqrt(w)
A = S*J; b = S*r; H=A.T*A; g=-A.T*b
```

先用原白化残差算 Huber，再以 `sqrt(w)` 同时缩放残差与雅可比；没有移动鲁棒阈值。验收比较同一固定初始有效支持上的真实 Huber cost，不用 IRLS 临时二次 cost 冒充目标函数。

以下任一情况保留原初估：初始求解未收敛或异常、初始投影支持不足、NaN/Inf、初始像素中位误差超过 8 像素或 log-depth 中位误差超过 0.5、可见证据不足、权重全低、有效样本量不足、空间支持集中、重估矩阵退化/条件数过大、不收敛、有效投影丢失、相对位姿跳变超限、原始像素中位误差明显变差，或固定支持的真实 Huber 目标变差。阈值见 YAML 和 `DEFAULTS`。

默认至少 64 个可见观测、20% 可见比例、64 个较高权重点及 10% 较高权重比例、至少覆盖 3×3 网格的 4 格；权重平均值至少 0.2，并检查有效样本量。重估以对角归一化的 7×7 正规矩阵特征值检查条件数，上限 `1e7`。更新限制为平移/中位深度≤0.1、旋转≤0.15 rad、log-scale≤0.1；固定支持不可投影即回退，投影数量也不得下降超过 2%。这些是保守首版参数，未通过真实数据标定。

原初估本身失败时仍按原跟踪失败/重定位逻辑处理；A 不制造一个可用初估。诊断写盘冲突或 I/O 失败也会保留初估并记录原因。CUDA 异步计时在 A 入口/出口 synchronize，因此 A 耗时含诊断开销，可能影响速度，不能声称保持实时。

## 日志、开销与检查边界

`A_CONFIG` 记录模式和生效参数；逐次 `A_EVENT` 记录源/目标帧、状态数量、权重均值/最小值/10/50/90 分位、支持、初始像素误差、重估步数/目标值（可用时）、A 耗时与累计次数。`attempts` 只计实际重估调用；`accepted` 计采纳；`fallbacks` 包括初估失败、跳过 A 和重估拒绝，诊断模式的无效证据也计入。

默认每 30 次 A 调用最多保存 8 组，每组 `.npz + .ppm`，输出由进程变量 `MAST3R_A_DIAGNOSTICS` 指向本次结果子目录。NPZ 有权重、有效性、状态、静态 UV、误差、`idx_f2k`、K 和初估相对矩阵。PPM 从左到右为状态、投影坐标颜色、特征误差、权重；未知区域须结合状态读，误差面板的填充值不是静态证据。不缓存跨轮特征，不新增共享张量或模型。

本机独立环境 `/home/banana/jushenzhinenggailun/work/mast3r_a_dev/.venv-cpu`，Python 安装也位于同项目 `.python/`。Python 3.11.18、torch 2.7.1+cpu、NumPy 1.26.4；完整列表和执行日志见 `evidence/`。Pillow/PyYAML 仅为检查已有接口使用，目标机没有新增依赖。

9 项 CPU 合成检查覆盖 Sim(3) 方向/尺度及雅可比、恒等投影、像素中心插值与索引、移动纹理不等于静态、无效/遮挡中性规则、原 Huber 顺序、权重对实际非线性位姿求解的影响、失败回退、与保存原 tracker 的关闭路径一致性、可选描述子接口调用次数、实际缩放裁剪 K 逻辑及配置协议、诊断只读/保存上限和不支持分支。CPU 用独立 `Sim3Double` 实现矩阵指数，检查生产 Python 几何/求解代码，**没有验证真实 lietorch 二进制**。另有 5 项交付夹具检查，验证增量补丁、复制排除、RealSense 保留、冲突/重复与失败处理。目标机的真实 spawn 导入、扩展内核、模型推理、完整 SLAM、ATE、显存与速度均待执行。

本机源码已恢复全部 Python/C++/CUDA/GLSL 和递归子模块；仅 `thirdparty/in3d/resources/meshes/mustard_bottle.ply` 这一独立示例网格因网络持续重置未取回，记录于 `guide-base/pre-development/resource-recovery.json`。它不是本次无界面 SLAM 的输入，未假装本机完整 GPU 工程已跑通。目标部署从实际可用基线复制该资源，不依赖本机副本。
