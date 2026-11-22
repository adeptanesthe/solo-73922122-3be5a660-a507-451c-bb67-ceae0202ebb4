"""训练切片规划 API 端到端测试：正常求解、无解、入参校验。"""
from __future__ import annotations

import copy

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

PATH = "/api/v1/training-slices"

SOLVED_PAYLOAD = {
    "image_width": 200,
    "image_height": 200,
    "ground_truths": [
        {"id": 1, "box": [40, 10, 60, 30]},
        {"id": 2, "box": [150, 10, 170, 30]},
    ],
    "candidate_crops": [
        [0, 0, 100, 100],
        [50, 0, 200, 100],
    ],
}


def post(payload):
    return client.post(PATH, json=payload)


def test_solved_response_shape_and_labels():
    resp = post(SOLVED_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "solved"
    assert body["selected_crop_indices"] == [0, 1]
    assert body["selected_crops"] == [[0, 0, 100, 100], [50, 0, 200, 100]]
    assert body["slice_count"] == 2
    assert body["optimal_solution_count"] >= 1
    assert body["uncovered_gt_ids"] == []
    assert body["blocked_components"] == []
    assert body["explanation"]

    s0, s1 = body["plan"]
    assert s0["crop_index"] == 0 and s0["crop"] == [0, 0, 100, 100]
    assert s0["positive_boxes"] == [{"gt_id": 1, "box": [40, 10, 60, 30]}]
    assert s0["ignore_boxes"] == []
    # 切片1 原点 (50,0)：GT2 正样本、GT1 被裁断部分为忽略
    assert s1["positive_boxes"] == [{"gt_id": 2, "box": [100, 10, 120, 30]}]
    assert s1["ignore_boxes"] == [{"gt_id": 1, "box": [0, 10, 10, 30]}]

    # 每个缺陷恰好一个正样本
    positive_ids = [b["gt_id"] for sp in body["plan"] for b in sp["positive_boxes"]]
    assert sorted(positive_ids) == [1, 2]


def test_infeasible_returns_empty_plan():
    payload = copy.deepcopy(SOLVED_PAYLOAD)
    # GT3 超出所有裁框
    payload["ground_truths"].append({"id": 3, "box": [190, 190, 199, 199]})
    resp = post(payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "infeasible"
    assert body["selected_crop_indices"] == []
    assert body["selected_crops"] == []
    assert body["plan"] == []
    assert body["slice_count"] == 0
    assert body["optimal_solution_count"] is None
    assert body["uncovered_gt_ids"] == [3]
    assert "调整" in body["explanation"]


def test_blocked_overlap_component_infeasible():
    payload = {
        "image_width": 100,
        "image_height": 100,
        "ground_truths": [
            {"id": 1, "box": [0, 0, 60, 60]},
            {"id": 2, "box": [50, 50, 90, 90]},
        ],
        "candidate_crops": [
            [0, 0, 70, 70],
            [40, 40, 100, 100],
        ],
    }
    body = post(payload).json()
    assert body["status"] == "infeasible"
    assert body["blocked_components"] == [[1, 2]]
    assert body["plan"] == []


def test_empty_ground_truths_yields_empty_plan():
    payload = copy.deepcopy(SOLVED_PAYLOAD)
    payload["ground_truths"] = []
    body = post(payload).json()
    assert body["status"] == "solved"
    assert body["plan"] == []
    assert body["slice_count"] == 0


# ---------------- 入参校验（422） ----------------

def test_float_coordinates_rejected():
    payload = copy.deepcopy(SOLVED_PAYLOAD)
    payload["candidate_crops"][0] = [0, 0, 100.0, 100]  # 100.0 不接受
    assert post(payload).status_code == 422


def test_bool_coordinate_rejected():
    payload = copy.deepcopy(SOLVED_PAYLOAD)
    payload["ground_truths"][0]["box"] = [40, 10, 60, True]
    assert post(payload).status_code == 422


def test_degenerate_box_rejected():
    payload = copy.deepcopy(SOLVED_PAYLOAD)
    payload["ground_truths"][0]["box"] = [40, 10, 40, 30]  # x2 == x1
    assert post(payload).status_code == 422


def test_too_many_ground_truths_rejected():
    payload = copy.deepcopy(SOLVED_PAYLOAD)
    payload["ground_truths"] = [
        {"id": i, "box": [i % 50, 0, i % 50 + 5, 5]} for i in range(61)
    ]
    assert post(payload).status_code == 422


def test_too_many_candidate_crops_rejected():
    payload = copy.deepcopy(SOLVED_PAYLOAD)
    payload["candidate_crops"] = [[0, 0, 10, 10]] * 13
    assert post(payload).status_code == 422


def test_crop_outside_image_rejected():
    payload = copy.deepcopy(SOLVED_PAYLOAD)
    payload["candidate_crops"].append([150, 0, 201, 100])  # x2 > image_width
    assert post(payload).status_code == 422


def test_gt_outside_image_rejected():
    payload = copy.deepcopy(SOLVED_PAYLOAD)
    payload["ground_truths"][0]["box"] = [40, 10, 60, 201]  # y2 > image_height
    assert post(payload).status_code == 422


def test_duplicate_gt_ids_rejected():
    payload = copy.deepcopy(SOLVED_PAYLOAD)
    payload["ground_truths"].append({"id": 1, "box": [10, 10, 20, 20]})
    assert post(payload).status_code == 422


def test_bad_box_length_rejected():
    payload = copy.deepcopy(SOLVED_PAYLOAD)
    payload["candidate_crops"].append([0, 0, 100])
    assert post(payload).status_code == 422


def test_non_positive_image_dimension_rejected():
    payload = copy.deepcopy(SOLVED_PAYLOAD)
    payload["image_width"] = 0
    assert post(payload).status_code == 422


def test_docs_page_available():
    resp = client.get("/docs")
    assert resp.status_code == 200
