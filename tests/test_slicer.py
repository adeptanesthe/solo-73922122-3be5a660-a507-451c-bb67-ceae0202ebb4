"""切片规划核心算法单元测试：包含关系、忽略区、最少切片搜索、无解诊断。"""
from __future__ import annotations

import random

from app.slicer import (
    SliceGroundTruth,
    clip_to,
    contains,
    plan_slices,
    positive_area_intersect,
)

G = SliceGroundTruth


def g(gid: int, box) -> SliceGroundTruth:
    return G(id=gid, box=tuple(box))


def solve(gts, crops, w=200, h=200):
    return plan_slices(list(gts), [tuple(c) for c in crops], w, h)


def by_gt(boxes):
    return {gid: box for gid, box in boxes}


# ---------------- 几何工具 ----------------

def test_positive_area_intersect_touching_edges():
    assert positive_area_intersect((0, 0, 10, 10), (5, 5, 15, 15))
    assert not positive_area_intersect((0, 0, 10, 10), (10, 0, 20, 10))  # 边相切
    assert not positive_area_intersect((0, 0, 10, 10), (20, 20, 30, 30))


def test_contains_and_clip():
    assert contains((0, 0, 100, 100), (10, 10, 20, 20))
    assert contains((0, 0, 10, 10), (0, 0, 10, 10))  # 边界重合算包含
    assert not contains((0, 0, 10, 10), (5, 5, 11, 11))
    assert clip_to((0, 0, 10, 10), (5, 5, 15, 15)) == (5, 5, 10, 10)
    assert clip_to((0, 0, 10, 10), (10, 0, 20, 10)) is None  # 零面积


# ---------------- 基础求解 ----------------

def test_single_crop_contains_all():
    gts = [g(1, (10, 10, 20, 20)), g(2, (30, 30, 40, 40))]
    crops = [(0, 0, 100, 100), (50, 50, 150, 150)]
    res = solve(gts, crops)
    assert res.status == "solved"
    assert res.selected_indices == [0]
    assert res.slice_count == 1
    assert res.optimal_solution_count == 1
    sp = res.plan[0]
    assert sp.crop_index == 0
    assert by_gt(sp.positive_boxes) == {1: (10, 10, 20, 20), 2: (30, 30, 40, 40)}
    assert sp.ignore_boxes == []


def test_minimum_number_of_slices_with_ignore_regions():
    # GT1 跨两个裁框，GT2 仅在第二个：最少需要 2 个切片
    gts = [g(1, (40, 10, 60, 30)), g(2, (150, 10, 170, 30))]
    crops = [(0, 0, 100, 100), (50, 0, 200, 100)]
    res = solve(gts, crops)
    assert res.status == "solved"
    assert res.selected_indices == [0, 1]
    assert res.slice_count == 2
    s0, s1 = res.plan
    # GT1 归属序号最小的可行切片 0（完整正样本）；切片 1 中露出部分为忽略框
    assert by_gt(s0.positive_boxes) == {1: (40, 10, 60, 30)}
    assert s0.ignore_boxes == []
    # 切片 1 原点 (50, 0)：GT1 交集 (50,10)-(60,30) → 局部 (0,10)-(10,30)
    assert by_gt(s1.positive_boxes) == {2: (100, 10, 120, 30)}
    assert by_gt(s1.ignore_boxes) == {1: (0, 10, 10, 30)}


def test_each_gt_is_positive_exactly_once():
    gts = [g(1, (10, 10, 90, 90)), g(2, (20, 20, 30, 30)), g(3, (150, 150, 160, 160))]
    crops = [(0, 0, 100, 100), (5, 5, 120, 120), (140, 140, 180, 180)]
    res = solve(gts, crops)
    assert res.status == "solved"
    positive_gt_ids = [gid for sp in res.plan for gid, _ in sp.positive_boxes]
    assert sorted(positive_gt_ids) == [1, 2, 3]
    assert len(positive_gt_ids) == 3  # 每个缺陷恰好一个正样本框
    # GT1/GT2 重叠 → 必须归属同一切片且该切片完整包含两者
    homes = {gid: sp.crop_index for sp in res.plan for gid, _ in sp.positive_boxes}
    assert homes[1] == homes[2]
    assert homes[3] == 2


def test_ignore_box_never_overlaps_positive_boxes_in_same_slice():
    # GT1 仅切片0 能完整包含；GT2 仅切片1 能完整包含；
    # 两 GT 相邻相切（x 边界 50 重合，零面积）。切片1 中 GT1 的露出为忽略框，
    # 与 GT2 正样本框无正面积重叠。
    gts = [g(1, (0, 0, 50, 50)), g(2, (50, 0, 90, 50))]
    crops = [(0, 0, 55, 60), (40, 0, 120, 60)]
    res = solve(gts, crops)
    assert res.status == "solved"
    assert res.selected_indices == [0, 1]
    s1 = next(sp for sp in res.plan if sp.crop_index == 1)
    pos = by_gt(s1.positive_boxes)
    ign = by_gt(s1.ignore_boxes)
    assert set(pos) == {2}
    assert set(ign) == {1}
    for pb in pos.values():
        for ib in ign.values():
            assert not positive_area_intersect(pb, ib)


def test_truncated_defect_generates_clipped_ignore_box():
    # GT1 归属切片0（完整包含）；切片1 裁到 GT1 右半部分 → 裁剪后的忽略框，
    # 保证被裁断缺陷不进背景样本
    gts = [g(1, (0, 0, 60, 60)), g(2, (120, 10, 140, 30))]
    crops = [(0, 0, 100, 100), (40, 0, 200, 100)]
    res = solve(gts, crops)
    assert res.status == "solved"
    s1 = next(sp for sp in res.plan if sp.crop_index == 1)
    # 切片1 原点 (40,0)：GT1 交集 (40,0)-(60,60) → 局部 (0,0)-(20,60)
    assert by_gt(s1.ignore_boxes) == {1: (0, 0, 20, 60)}
    assert by_gt(s1.positive_boxes) == {2: (80, 10, 100, 30)}


def test_ignore_whole_box_when_fully_visible_in_other_slice():
    # 两个切片都必需：GT3 仅 crop0 能完整包含；GT2 仅 crop1 能完整包含；
    # GT1 在两个切片中都完整可见 → 归属切片0（序号最小），切片1 中为整框忽略
    gts = [
        g(1, (10, 10, 20, 20)),
        g(2, (80, 80, 90, 90)),
        g(3, (0, 60, 10, 70)),
    ]
    crops = [(0, 0, 50, 100), (5, 5, 100, 100)]
    res = solve(gts, crops)
    assert res.selected_indices == [0, 1]
    s0 = next(sp for sp in res.plan if sp.crop_index == 0)
    s1 = next(sp for sp in res.plan if sp.crop_index == 1)
    # GT1 归属序号最小的可行切片 0
    assert dict(s0.positive_boxes)[1] == (10, 10, 20, 20)
    # 切片1 原点 (5,5)：GT1 完整可见，忽略框为整框局部 (5,5)-(15,15)；
    # GT3 被 crop1 裁断（交集 (5,60)-(10,70)）→ 局部忽略框 (0,55)-(5,65)
    assert by_gt(s1.ignore_boxes) == {
        1: (5, 5, 15, 15),
        3: (0, 55, 5, 65),
    }
    assert dict(s1.positive_boxes)[2] == (75, 75, 85, 85)


def test_edge_touch_does_not_create_ignore_box():
    # 两个切片都必需；GT1 与切片1 仅边相切（x2=50 == 切片 x1=50）：不产生忽略框
    gts = [g(1, (10, 10, 50, 50)), g(2, (60, 10, 80, 50))]
    crops = [(0, 0, 55, 60), (50, 0, 100, 60)]
    res = solve(gts, crops)
    assert res.selected_indices == [0, 1]
    s1 = next(sp for sp in res.plan if sp.crop_index == 1)
    assert 1 not in dict(s1.ignore_boxes)
    assert set(dict(s1.positive_boxes)) == {2}


def test_overlapping_gts_must_share_home_slice():
    # 两个重叠缺陷分别只有不同裁框能完整包含 → 无解（阻塞组）
    gts = [g(1, (0, 0, 60, 60)), g(2, (50, 50, 90, 90))]
    crops = [(0, 0, 70, 70), (40, 40, 100, 100)]
    # crop0 完整包含 GT1 但装不下 GT2；crop1 反之
    res = solve(gts, crops)
    assert res.status == "infeasible"
    assert res.selected_indices == []
    assert res.plan == []
    assert res.optimal_solution_count is None
    assert res.blocked_components == [[1, 2]]
    assert "调整" in res.explanation


def test_uncovered_gt_is_infeasible():
    gts = [g(1, (0, 0, 10, 10)), g(2, (95, 95, 105, 105))]
    crops = [(0, 0, 100, 100)]  # GT2 超出裁框，无裁框完整包含
    res = solve(gts, crops)
    assert res.status == "infeasible"
    assert res.uncovered_gt_ids == [2]
    assert "2" in res.explanation


def test_no_crops_with_gts_is_infeasible():
    res = solve([g(1, (0, 0, 10, 10))], [])
    assert res.status == "infeasible"
    assert res.uncovered_gt_ids == [1]
    assert res.plan == []


def test_empty_gts_returns_empty_plan():
    res = solve([], [(0, 0, 100, 100)])
    assert res.status == "solved"
    assert res.selected_indices == []
    assert res.plan == []
    assert res.slice_count == 0
    assert res.optimal_solution_count == 1


def test_tie_break_prefers_smallest_crop_index():
    # 两个裁框都能独立满足，同属最少（1 片）：取候选序号最小者
    gts = [g(1, (10, 10, 20, 20))]
    crops = [(5, 5, 30, 30), (0, 0, 100, 100)]
    res = solve(gts, crops)
    assert res.selected_indices == [0]
    assert res.optimal_solution_count == 2


def test_duplicate_crops_count_as_separate_solutions():
    gts = [g(1, (10, 10, 20, 20))]
    crops = [(0, 0, 100, 100), (0, 0, 100, 100)]
    res = solve(gts, crops)
    assert res.status == "solved"
    assert res.selected_indices == [0]
    assert res.optimal_solution_count == 2  # 选 0 或选 1 都是最少方案


def test_partial_intersection_does_not_satisfy_containment():
    # 裁框只盖住缺陷的一部分：不能作为正样本归属，且无其他裁框 → 无解
    gts = [g(1, (40, 40, 80, 80))]
    crops = [(0, 0, 60, 60)]
    res = solve(gts, crops)
    assert res.status == "infeasible"
    assert res.uncovered_gt_ids == [1]
    assert res.plan == []


def test_overlap_component_three_gts_common_crop_required():
    # GT1∩GT2、GT2∩GT3 连成一个连通分量，必须三者同一切片
    gts = [
        g(1, (0, 0, 40, 40)),
        g(2, (30, 30, 70, 70)),
        g(3, (60, 60, 90, 90)),
    ]
    # crop0 装下前两个，crop1 装下后两个，只有 crop2 装下全部
    crops = [(0, 0, 75, 75), (25, 25, 95, 95), (0, 0, 100, 100)]
    res = solve(gts, crops)
    assert res.status == "solved"
    assert res.selected_indices == [2]
    homes = {gid: sp.crop_index for sp in res.plan for gid, _ in sp.positive_boxes}
    assert homes == {1: 2, 2: 2, 3: 2}


def test_local_coordinates_use_slice_origin():
    gts = [g(7, (20, 30, 40, 50))]
    crops = [(10, 20, 100, 100)]
    res = solve(gts, crops)
    sp = res.plan[0]
    assert by_gt(sp.positive_boxes) == {7: (10, 10, 30, 30)}  # 减去 (10,20)


def test_randomized_invariants():
    """随机用例下的全局不变量：
    1. 有解时每个缺陷恰好一个正样本框，且正样本框=缺陷原图框（经原点平移）；
    2. 忽略框与同一切片内任何正样本框无正面积重叠；
    3. 忽略框正是非归属缺陷与切片的正面积交集；
    4. 局部框平移回原图后落在切片范围内；
    5. 选中数量确实是最小的（与逐子集枚举的最小尺寸一致）。
    """
    rng = random.Random(20261122)
    W = H = 300
    for _ in range(300):
        n_gt = rng.randint(0, 12)
        gts = []
        for gid in range(n_gt):
            x1 = rng.randint(0, W - 10)
            y1 = rng.randint(0, H - 10)
            gts.append(g(gid, (x1, y1, x1 + rng.randint(2, 12), y1 + rng.randint(2, 12))))
        n_crop = rng.randint(0, 8)
        crops = []
        for _ in range(n_crop):
            x1 = rng.randint(0, W - 20)
            y1 = rng.randint(0, H - 20)
            crops.append((x1, y1, x1 + rng.randint(20, 80), y1 + rng.randint(20, 80)))

        res = plan_slices(gts, crops, W, H)

        # 独立计算最小可行尺寸做交叉验证
        def min_feasible_size() -> int | None:
            from itertools import combinations as comb
            for size in range(n_crop + 1):
                for idxs in comb(range(n_crop), size):
                    sub = set(idxs)
                    ok = True
                    seen: dict[int, int] = {}
                    for k, gt in enumerate(gts):
                        opts = [c for c in sub if contains(crops[c], gt.box)]
                        # 重叠约束：与已归属缺陷重叠时必须同一切片
                        valid = [c for c in opts
                                 if all(c == seen[j] for j in seen
                                        if positive_area_intersect(gt.box, gts[j].box))]
                        if not valid:
                            ok = False
                            break
                        seen[k] = min(valid)
                    if ok:
                        return size
            return None

        expected_size = min_feasible_size()
        if expected_size is None:
            assert res.status == "infeasible"
            assert res.plan == [] and res.selected_indices == []
            continue

        assert res.status == "solved"
        assert len(res.selected_indices) == expected_size
        homes: dict[int, int] = {}
        for sp in res.plan:
            crop = sp.crop
            ox, oy = crop[0], crop[1]
            pos = dict(sp.positive_boxes)
            ign = dict(sp.ignore_boxes)
            for gid, box in sp.positive_boxes:
                assert gid not in homes  # 每个缺陷恰好一个正样本
                homes[gid] = sp.crop_index
                gt_box = dict((x.id, x.box) for x in gts)[gid]
                assert box == (gt_box[0] - ox, gt_box[1] - oy,
                               gt_box[2] - ox, gt_box[3] - oy)
                assert contains(crop, gt_box)
            for gid, box in sp.ignore_boxes:
                # 局部框在切片范围内
                assert 0 <= box[0] < box[2] <= crop[2] - crop[0]
                assert 0 <= box[1] < box[3] <= crop[3] - crop[1]
                gt_box = dict((x.id, x.box) for x in gts)[gid]
                part = clip_to(crop, gt_box)
                assert part is not None
                assert box == (part[0] - ox, part[1] - oy,
                               part[2] - ox, part[3] - oy)
                # 忽略框与同切片所有正样本框无正面积重叠
                for pb in pos.values():
                    assert not positive_area_intersect(pb, box)
            # 忽略框集合恰好对应非归属且与本切片正面积相交的缺陷
            expected_ign = {
                x.id for x in gts
                if homes.get(x.id, sp.crop_index) != sp.crop_index
                and clip_to(crop, x.box) is not None
            }
            assert set(ign) == expected_ign
        assert set(homes) == set(range(n_gt))
