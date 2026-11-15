---
name: aorta-segment
description: "Run the aorta-segment pipeline: one aortic CTA case directory to a fused vessel tree (mesh + STEP + viewer + per-segment acceptance). Use when the user asks 主动脉分割、血管树重建、CTA 到网格、主动脉分段、segment 工具、aorta segment, or to re-run or verify an existing case."
---

# aorta-segment

## 何时用

用户要把一例主动脉 CTA（`ct.nii.gz` + `crop_meta.json`）变成可进 CFD/FEA 的融合血管树，
或要复核既有 `out_pipeline/` 的逐段验收表。

## 命令

```bash
python seg.py check          # 依赖、Slicer、TotalSegmentator 许可、模型落点
python seg.py run  <case>    # 单例；--no-ts 复用上次分割，--extra name=file.nii.gz 接入额外血管
python seg.py batch <root>   # 批量
python seg.py info <case>    # 产物与验收
python seg.py render <case>  # PNG 预览
```

`SLICER_PY` 指向 Slicer 的 `PythonSlicer`；未设置时自动探测。

## 硬约束（随工具一起走）

- 验收门槛按段设定：主动脉约 171 mL、腹腔干约 1.6 mL、单支冠脉约 0.14 mL，跨三个数量级，不能用一个门槛套所有段。
- 短血管不用中轴长度判「短桩」：腹腔干中轴天然 10–15 mm。
- 半径用法平面切割；主干取它与内切球的较小者（斜切会放大弯管）。
- 布尔并集会在接头留领口；必须用同一栅格取或。
- 每次运行写 `manifest.json`（版本、模型、许可状态、输入哈希、分段表），结果可追溯到配置。

## 边界

研究用途，不是医疗器械。仓库与运行产物不得包含患者数据；病例由使用者自备。
