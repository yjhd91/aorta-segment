# aorta-segment

从一例主动脉 CTA 到一棵**融合血管树**：网格 + STEP + 可拖动查看器 + 逐段验收表，一条命令。

One command from an aortic CTA case directory to a fused vessel tree: mesh, STEP, an orbit viewer and a
per-segment acceptance table.

```
python seg.py check          # 本机能否运行
python seg.py run  <case>    # 单例 -> <case>/out_pipeline/
python seg.py batch <root>   # 批量
python seg.py info <case>    # 查看产物与验收
python seg.py render <case>  # PNG 预览
```

## 病例目录

```
<case>/
  ct.nii.gz          CTA 体数据
  crop_meta.json     {"sp_zyx": [z,y,x 间距], "org_c": [x,y,z 原点], "crop": [[k0,k1],[i0,i1],[j0,j1]]}
```

## 流水线

1. **分割** TotalSegmentator 出主动脉主干、弓上三支、双髂；腹腔干/肾动脉由器官最小代价路径追踪，
   SMA 由向下追踪得到。任何额外血管（冠脉、变异支）可用 `--extra name=mask.nii.gz` 接入。
2. **中心线** 每支的中轴，按图直径排序（不用最近邻行走：会锯齿并把扫掠管撑大约 50%）。
3. **半径** 法平面切割面积；主干取它与内切球乘椭圆度因子的较小者（斜切平面会放大弯管）。
4. **口径** 每段半径曲线整体缩放，使扫掠管体积与掩膜体积精确相等。
5. **分叉** 分支以等半径延伸过主干中心线；主干在最后 25 mm 收细到 `sqrt(rL²+rR²)`，双髂从此点出发。
6. **并集** 所有管在 0.6 mm 栅格上取或（布尔并集会在每个接头留领口）。
7. **表面** 有符号距离场模糊 1.8 mm 后取零等值面，圆化凹折；清理并轻量 Taubin 平滑。
8. **交付** `model.stl`、`model.step`、自包含 orbit 查看器、`acceptance.json`（逐段体积/半径/连通性）。

## 依赖

- Python 3.10+：`numpy scipy scikit-image trimesh nibabel`（见 `requirements.txt`）
- 3D Slicer 5.12.4 + TotalSegmentator（分割阶段在 Slicer 的 Python 里跑）
- `SLICER_PY` 环境变量可覆盖 Slicer 的 `PythonSlicer` 路径；未设置时按常见安装位置探测

## 边界

- 研究用途，不是医疗器械，不提供临床决策。
- 仓库不含任何患者数据或掩膜；病例目录由使用者自备。
- 分割阶段调用 TotalSegmentator；本仓库不包含任何商业产品的代码或复现。

## 文档

- [`docs/SEG_TOOL.md`](docs/SEG_TOOL.md)：`seg.py` 的命令、产物与验收口径
- [`docs/AORTA_PIPELINE.md`](docs/AORTA_PIPELINE.md)：各阶段实现与为什么这样选
- [`skill/SKILL.md`](skill/SKILL.md)：给 AI agent 的技能入口（DSH / Claude Code / Codex）

## License

MIT，见 [`LICENSE`](LICENSE)。
