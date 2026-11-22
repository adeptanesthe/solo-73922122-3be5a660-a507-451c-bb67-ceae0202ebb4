# 釉面缺陷检测辅助 API

本服务提供两组能力：

1. **漏检复核**：相邻的釉面缺陷在 NMS 后可能只剩一个框，造成漏检。本服务在
   **抑制前**的预测框上重放「贪心 NMS + 贪心匹配」，并搜索：为找回指定漏检真值，
   最少需要豁免多少对预测框之间的抑制。
2. **训练切片规划**（新增）：从候选裁框中选出满足标注规则的**最少切片**，
   完整缺陷作为正样本框，被裁断/露出的缺陷标为忽略区（不进背景样本）。

## 规则约定

1. **NMS**：预测框按置信度降序（分数并列取编号较小者）做贪心 NMS；只有被保留
   的框能抑制 IoU **严格大于** `iou_threshold` 的后续框。
2. **匹配**：按同一顺序，把每个保留框配给 IoU 最大且 **≥ 0.5** 的未配真值；
   IoU 并列取编号较小的真值。未配上的保留框计为**误报**。
3. **豁免**：一对豁免框仅解除这两个框之间的相互抑制，不影响其他框对；被释放
   的框之后同样可以抑制别人（可能产生级联）。
4. **目标**：找到**最少**的豁免对，使得
   - 指定的待找回真值全部命中；
   - 无豁免时原有命中的真值（按真值编号计，允许更换命中它的预测框）全部保留；
   - 误报数不超过无豁免时。
   同为最少的方案有多个时，返回按候选对顺序的第一个，并在
   `optimal_solution_count` 中给出方案总数。
5. 候选对中两框 IoU 未超过抑制阈值的，豁免无意义，会被列入
   `ignored_ineffective_pairs` 而不参与搜索。

## 限制

- 预测框、真值框各至多 **60** 个；可豁免候选框对至多 **10** 对。
- 框格式 `[x1, y1, x2, y2]`，要求 `x2>x1`、`y2>y1`；置信度 ∈ [0, 1]；
  `iou_threshold` ∈ [0, 1]；编号在同一张图内唯一（非负整数）。

## 启动（Docker Compose）

```bash
docker compose up --build
# 服务监听 http://localhost:8000 ，交互式文档见 http://localhost:8000/docs
```

## 本地开发

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/uvicorn app.main:app --reload
.venv/bin/pytest -q          # 运行测试
```

## API

### `GET /health`

健康检查，返回 `{"status": "ok"}`。

### `POST /api/v1/review`

请求体：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `predictions` | array | 抑制前预测框 `[{id, box, score}]`，≤60 |
| `ground_truths` | array | 真值框 `[{id, box}]`，≤60 |
| `target_gt_ids` | int[] | 待找回的真值编号 |
| `iou_threshold` | float | NMS 抑制阈值（IoU 严格大于它才抑制） |
| `exemptible_pairs` | int[][] | 可豁免的预测框编号对，≤10 对 |

响应要点：

- `status`: `solved` / `infeasible`
- `min_exemption_count` / `chosen_exempt_pairs`: 最少豁免对数与选中的框对
- `baseline`: 无豁免时的 NMS+匹配明细（保留框、被抑制框、配对、误报、漏检）
- `after`: 采用最少豁免后的同样明细（无解时为 `null`）
- `all_exempted`: 豁免全部有效候选对后的明细（仅无解时给出，用于诊断）
- `explanation`: 结论说明；无解时列出未满足的条件

### 可调用示例

启动服务后：

```bash
# 场景一：两个相邻缺陷，P2 命中 GT102 但被 P1 抑制
curl -sS -X POST http://localhost:8000/api/v1/review \
  -H 'Content-Type: application/json' \
  -d @examples/request_solved.json
```

关键响应（完整字段略）：

```json
{
  "status": "solved",
  "min_exemption_count": 1,
  "chosen_exempt_pairs": [[1, 2]],
  "explanation": "最少需豁免 1 对预测框：(1,2)。豁免后指定漏检全部找回，原有命中全部保留，误报数 1 → 1（未增加）。",
  "baseline":  { "kept_prediction_ids": [1, 3], "missed_gt_ids": [102], "false_positive_count": 1 },
  "after":     { "kept_prediction_ids": [1, 2, 3], "hit_gt_ids": [101, 102], "false_positive_count": 1 }
}
```

```bash
# 场景二：GT103 无任何预测框覆盖，豁免也找不回 → 无解并说明原因
curl -sS -X POST http://localhost:8000/api/v1/review \
  -H 'Content-Type: application/json' \
  -d @examples/request_infeasible.json
```

```json
{
  "status": "infeasible",
  "min_exemption_count": null,
  "explanation": "无解：候选豁免对的任何子集都无法同时满足「找回指定漏检、原有命中保留、误报不增」。指定真值 [103] 在豁免全部 1 对有效候选框后仍未命中。"
}
```

也可以直接运行 `examples/call_api.sh`（服务需已启动）：

```bash
./examples/call_api.sh                                  # 场景一
./examples/call_api.sh http://localhost:8000 examples/request_infeasible.json   # 场景二
```

### `POST /api/v1/training-slices`

为釉面缺陷检测器制作训练切片：从候选裁框中选出满足标注规则的**最少切片**，
被裁断的缺陷以忽略区形式排除出背景样本。

**标注规则**

1. 每个缺陷必须至少被一个入选切片**完整包含**，并恰好指定一个「归属切片」
   保留完整正样本框；同一缺陷在其他入选切片中的露出部分（完整露出或被裁断）
   一律标为**忽略框**，仅正面积相交才生成（边相切不生成）。
2. 忽略框与同一切片内的正样本框不得有正面积重叠。由此，凡有正面积重叠的缺陷
   必须归属同一切片，且该切片必须完整包含整组重叠缺陷。
3. 无解（某缺陷无裁框完整包含，或某重叠缺陷组无法共置于同一裁框）时返回空计划，
   并在 `explanation` 中提示调整候选裁框。

**限制**：原图宽高为正整数；真值框至多 **60** 个，候选裁框至多 **12** 个；
全部坐标为整数像素；框格式 `[x1, y1, x2, y2]`，要求 `x2>x1`、`y2>y1`
且不超出原图；真值编号在同一张图内唯一。

请求体：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `image_width` | int | 原图宽度（整数像素） |
| `image_height` | int | 原图高度（整数像素） |
| `ground_truths` | array | 缺陷真值框 `[{id, box}]`，≤60 |
| `candidate_crops` | int[][] | 候选裁框 `[x1, y1, x2, y2]`，≤12 |

响应要点：

- `status`: `solved` / `infeasible`（无解时计划为空）
- `selected_crop_indices` / `selected_crops` / `slice_count`: 入选裁框序号、原图坐标与数量
- `plan`: 每个入选切片的 `crop_index`、`crop` 以及
  - `positive_boxes`: 以切片左上角为原点的完整正样本框 `[{gt_id, box}]`（每缺陷恰好一个）
  - `ignore_boxes`: 同坐标系的忽略框 `[{gt_id, box}]`（与正样本框无正面积重叠）
- `optimal_solution_count`: 同为最少切片数的可行方案总数；并列时返回按候选裁框顺序的第一个
- `uncovered_gt_ids` / `blocked_components`: 无解诊断（无裁框完整包含的缺陷、
  无法共置同一裁框的重叠缺陷组）
- `explanation`: 结论说明；无解时提示如何调整候选裁框

```bash
# 有解：GT1 跨两个切片，在切片1 中被裁断的部分标为忽略框
curl -sS -X POST http://localhost:8000/api/v1/training-slices \
  -H 'Content-Type: application/json' \
  -d @examples/slice_request_solved.json
```

关键响应：

```json
{
  "status": "solved",
  "selected_crop_indices": [0, 1],
  "slice_count": 2,
  "plan": [
    {
      "crop_index": 0, "crop": [0, 0, 100, 100],
      "positive_boxes": [{"gt_id": 1, "box": [40, 10, 60, 30]}],
      "ignore_boxes": []
    },
    {
      "crop_index": 1, "crop": [50, 0, 200, 100],
      "positive_boxes": [{"gt_id": 2, "box": [100, 10, 120, 30]}],
      "ignore_boxes": [{"gt_id": 1, "box": [0, 10, 10, 30]}]
    }
  ],
  "optimal_solution_count": 1,
  "explanation": "选中 2 个切片：候选裁框序号 [0, 1]。……共 2 个正样本框、1 个忽略框。"
}
```

```bash
# 无解：两个相互重叠的缺陷没有任何候选裁框能同时完整装下
curl -sS -X POST http://localhost:8000/api/v1/training-slices \
  -H 'Content-Type: application/json' \
  -d @examples/slice_request_infeasible.json
```

```json
{
  "status": "infeasible",
  "selected_crop_indices": [],
  "plan": [],
  "slice_count": 0,
  "optimal_solution_count": null,
  "blocked_components": [[1, 2]],
  "explanation": "无解：现有候选裁框无法满足标注规则。相互重叠的缺陷组 [[1, 2]] 无法同时完整落入同一候选裁框。请调整候选裁框（增大裁框、补充覆盖或重新排布）后重试。"
}
```

也可以直接运行 `examples/call_slicing_api.sh`（服务需已启动）：

```bash
./examples/call_slicing_api.sh                                          # 有解示例
./examples/call_slicing_api.sh http://localhost:8000 examples/slice_request_infeasible.json
```

## 实现说明

- 漏检复核核心逻辑在 `app/solver.py`：先一次性预计算预测框两两 IoU 与预测框-真值 IoU
  矩阵，再对候选对的每个子集（≤2¹⁰）重放 NMS+匹配，按豁免对数从小到大找到
  第一个可行子集即为最优解；无解时用「全部豁免」的结果生成诊断说明。
- 切片规划核心逻辑在 `app/slicer.py`：
  - 预计算每个缺陷能被哪些裁框完整包含，并按缺陷框的正面积重叠关系（并查集）
    求连通分量——同分量缺陷必须归属同一个完整包含整组的裁框（忽略框/正样本框
    不重叠约束的等价条件）；
  - 对候选裁框子集按切片数从小到大枚举（≤2¹²=4096），每个分量取序号最小的
    可行裁框作为归属，找到的第一个可行子集即最少切片方案，并统计同为最少的方案数；
  - 构造结果时，归属切片输出平移到切片局部坐标的完整正样本框；其余切片中与该缺陷
    正面积相交的区域（含被裁断部分）输出裁剪后的局部忽略框；
  - 无解时区分「无裁框完整包含」与「重叠缺陷组无法共置」两类诊断，提示调整裁框。
- 校验在 `app/models.py`（Pydantic）：数量上限、整数像素（`before` 校验器拒绝
  布尔/浮点，避免宽松强转）、框合法性、编号唯一、框不超出原图等，
  不合法请求返回 422 及中文错误信息。
- 测试在 `tests/`：漏检复核 35 个用例；切片规划 34 个用例，覆盖包含/裁切几何、
  忽略区与正样本框不重叠、边相切不生成忽略框、重叠缺陷共置约束、最少切片与并列
  计数、无解两类诊断、全部 422 分支，以及 300 组随机用例对照独立暴力枚举的
  不变量校验；另含 API 端到端与 `/docs` 可用性。
