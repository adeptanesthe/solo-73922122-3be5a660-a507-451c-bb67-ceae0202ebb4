"""训练切片规划核心算法。

需求（与标注规则一致）：
1. 给定原图尺寸、至多 60 个缺陷真值框、至多 12 个候选裁框（整数像素）。
2. 选出**最少**的候选裁框（切片），使每个缺陷都至少被一个入选切片**完整包含**；
   每个缺陷恰好指定一个「归属切片」作为其正样本框（完整框）所在的切片。
3. 同一缺陷在其他入选切片中若有露出，则该露出部分标为**忽略区**（被裁断的缺陷
   因此不会落入背景样本）。仅正面积相交才产生忽略框；边相切（零面积）不产生。
4. 忽略区与同一切片内的正样本框不得有正面积重叠。由此推出：凡有正面积重叠的
   缺陷必须归属同一切片，且该切片必须完整包含整组重叠缺陷。
5. 无解（某缺陷无裁框完整包含，或某重叠缺陷组无法共置于同一裁框）时返回空计划，
   并给出诊断信息，提示调整候选裁框。

坐标全部为整数像素；框格式沿用 [x1, y1, x2, y2]。
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

MAX_GROUND_TRUTHS = 60
MAX_CANDIDATE_CROPS = 12

Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class SliceGroundTruth:
    id: int
    box: Box


@dataclass
class SlicePlan:
    """单个入选切片的规划结果（坐标均以该切片左上角为原点）。"""

    crop_index: int
    crop: Box                                  # 裁框在原图中的坐标
    positive_boxes: list[tuple[int, Box]]      # (缺陷编号, 局部正样本框)
    ignore_boxes: list[tuple[int, Box]]        # (缺陷编号, 局部忽略框)


@dataclass
class SlicePlanResult:
    status: str                                # "solved" | "infeasible"
    selected_indices: list[int]                # 入选裁框的候选序号（0 起）
    plan: list[SlicePlan]
    optimal_solution_count: int | None         # 同为最少切片数的可行方案个数（无解为 None）
    uncovered_gt_ids: list[int]                # 无任何裁框完整包含的缺陷
    blocked_components: list[list[int]]        # 重叠但无法共置同一裁框的缺陷组
    explanation: str

    @property
    def slice_count(self) -> int:
        return len(self.selected_indices)


def positive_area_intersect(a: Box, b: Box) -> bool:
    """两框是否存在正面积交集（边相切不算）。"""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    return iw > 0 and ih > 0


def contains(outer: Box, inner: Box) -> bool:
    """outer 是否完整包含 inner（边界重合允许）。"""
    return (outer[0] <= inner[0] and outer[1] <= inner[1]
            and outer[2] >= inner[2] and outer[3] >= inner[3])


def clip_to(a: Box, b: Box) -> Box | None:
    """两框交集；无正面积交集时返回 None。"""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 > x1 and y2 > y1:
        return (x1, y1, x2, y2)
    return None


def _connected_components(gts: list[SliceGroundTruth]) -> list[list[int]]:
    """按正面积重叠关系求缺陷连通分量（并查集）。"""
    parent = list(range(len(gts)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(gts)):
        for j in range(i + 1, len(gts)):
            if positive_area_intersect(gts[i].box, gts[j].box):
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(len(gts)):
        groups.setdefault(find(i), []).append(i)
    return [sorted(v) for v in groups.values()]


def plan_slices(
    gts: list[SliceGroundTruth],
    crops: list[Box],
    image_width: int,
    image_height: int,
) -> SlicePlanResult:
    """搜索满足标注规则的最少切片方案。

    image_width / image_height 仅用于文档化与上层校验，几何上不参与计算
    （入参合法性由 Pydantic 模型保证）。
    """
    m, n = len(gts), len(crops)

    # 1. 每个缺陷能被哪些裁框完整包含
    containing: list[list[int]] = [
        [c for c in range(n) if contains(crops[c], gts[g].box)]
        for g in range(m)
    ]
    uncovered = sorted(
        gts[g].id for g in range(m) if not containing[g]
    )

    # 2. 按重叠关系分组；同组缺陷必须归属同一个（完整包含整组的）裁框
    components = _connected_components(gts)
    home_options: list[list[int]] = []
    blocked_components: list[list[int]] = []
    for members in components:
        common = set(containing[members[0]])
        for g in members[1:]:
            common &= set(containing[g])
        home_options.append(sorted(common))
        if not common and all(containing[g] for g in members):
            # 每个缺陷单独都有裁框可放，但没有一个裁框能完整装下整组重叠缺陷
            blocked_components.append(sorted(gts[g].id for g in members))

    # 3. 子集枚举（n <= 12，至多 2^12 = 4096 个子集），按切片数从少到多
    def homes_for(subset: frozenset[int]) -> dict[int, int] | None:
        """返回该子集下每个缺陷的归属裁框；不可行返回 None。

        并列时取候选序号最小的裁框，保证结果确定。
        """
        homes: dict[int, int] = {}
        for members, options in zip(components, home_options):
            pick = next((c for c in options if c in subset), None)
            if pick is None:
                return None
            for g in members:
                homes[g] = pick
        return homes

    first_combo: tuple[int, ...] | None = None
    first_homes: dict[int, int] | None = None
    feasible_count = 0
    for size in range(0, n + 1):
        for combo in combinations(range(n), size):
            homes = homes_for(frozenset(combo))
            if homes is not None:
                feasible_count += 1
                if first_combo is None:
                    first_combo = combo
                    first_homes = homes
        if first_combo is not None:
            break

    if first_combo is None or first_homes is None:
        problems: list[str] = []
        if uncovered:
            problems.append(f"缺陷 {uncovered} 无法被任一候选裁框完整包含")
        if blocked_components:
            problems.append(
                f"相互重叠的缺陷组 {blocked_components} 无法同时完整落入同一候选裁框"
            )
        if not problems:  # 兜底（例如未提供任何裁框）
            problems.append("候选裁框无法覆盖全部缺陷")
        explanation = (
            "无解：现有候选裁框无法满足标注规则。"
            + "；".join(problems)
            + "。请调整候选裁框（增大裁框、补充覆盖或重新排布）后重试。"
        )
        return SlicePlanResult(
            status="infeasible",
            selected_indices=[],
            plan=[],
            optimal_solution_count=None,
            uncovered_gt_ids=uncovered,
            blocked_components=blocked_components,
            explanation=explanation,
        )

    # 4. 构造入选切片的正样本框 / 忽略框（局部坐标）
    plan: list[SlicePlan] = []
    total_positive = 0
    total_ignore = 0
    for s in first_combo:
        crop = crops[s]
        ox, oy = crop[0], crop[1]
        positives: list[tuple[int, Box]] = []
        ignores: list[tuple[int, Box]] = []
        for g in range(m):
            gt = gts[g]
            if first_homes[g] == s:
                positives.append((gt.id, (
                    gt.box[0] - ox, gt.box[1] - oy,
                    gt.box[2] - ox, gt.box[3] - oy,
                )))
            else:
                # 归属其他切片的缺陷在本切片的露出部分（完整露出或被裁断）
                # 一律标为忽略区，确保被裁断缺陷不落入背景样本
                part = clip_to(crop, gt.box)
                if part is not None:
                    ignores.append((gt.id, (
                        part[0] - ox, part[1] - oy,
                        part[2] - ox, part[3] - oy,
                    )))
        positives.sort(key=lambda x: x[0])
        ignores.sort(key=lambda x: x[0])
        total_positive += len(positives)
        total_ignore += len(ignores)
        plan.append(SlicePlan(
            crop_index=s, crop=crop,
            positive_boxes=positives, ignore_boxes=ignores,
        ))

    selected = list(first_combo)
    if m == 0:
        explanation = "无真值框：无需切片，返回空计划。"
    else:
        explanation = (
            f"选中 {len(selected)} 个切片：候选裁框序号 {selected}。"
            "每个缺陷恰好在一个入选切片中保留完整正样本框，"
            "其在其他入选切片中的露出部分标为忽略区，"
            "且忽略区与同一切片内的正样本框无正面积重叠。"
            f"共 {total_positive} 个正样本框、{total_ignore} 个忽略框。"
        )
        if feasible_count > 1:
            explanation += (
                f"（另有 {feasible_count - 1} 个同为最少切片数的可行方案，"
                "返回按候选裁框顺序的第一个。）"
            )

    return SlicePlanResult(
        status="solved",
        selected_indices=selected,
        plan=plan,
        optimal_solution_count=feasible_count,
        uncovered_gt_ids=[],
        blocked_components=[],
        explanation=explanation,
    )
