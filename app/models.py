"""请求 / 响应数据模型与入参校验。"""
from __future__ import annotations

import math

from pydantic import BaseModel, Field, field_validator, model_validator

MAX_BOXES = 60          # 预测框、真值框各至多 60 个
MAX_EXEMPT_PAIRS = 10   # 可豁免候选框对至多 10 对
MAX_SLICE_GTS = 60      # 切片规划：真值框至多 60 个
MAX_CANDIDATE_CROPS = 12  # 切片规划：候选裁框至多 12 个


def _check_finite(coords: list[float]) -> list[float]:
    if any(not math.isfinite(c) for c in coords):
        raise ValueError("坐标/分数必须是有限数值")
    return coords


def _check_int_box(box: object) -> object:
    """整数像素框校验（before 模式，抢在 Pydantic 宽松整数强转之前）。

    拒绝布尔与浮点（含 1.0），要求 4 个整数且 x2>x1、y2>y1。
    """
    if not isinstance(box, list):
        return box  # 交给 Pydantic 报类型/长度错误
    for c in box:
        if isinstance(c, bool) or not isinstance(c, int):
            raise ValueError("坐标必须是整数像素（不接受布尔值或浮点数，如 1.0）")
    if len(box) == 4 and not (box[2] > box[0] and box[3] > box[1]):
        raise ValueError("box 必须满足 x2>x1 且 y2>y1（格式 [x1, y1, x2, y2]）")
    return box


def _check_int(v: object) -> object:
    """整数字段校验（before 模式）：拒绝布尔与浮点。"""
    if isinstance(v, bool) or (isinstance(v, float) and v.is_integer()):
        raise ValueError("必须是整数（不接受布尔值或浮点数，如 1.0）")
    if not isinstance(v, int):
        raise ValueError("必须是整数")
    return v


class PredictionIn(BaseModel):
    id: int = Field(ge=0, description="预测框编号（同一张图内唯一）")
    box: list[float] = Field(min_length=4, max_length=4,
                             description="[x1, y1, x2, y2]，要求 x2>x1 且 y2>y1")
    score: float = Field(ge=0.0, le=1.0, description="置信度，取值 [0, 1]")

    @field_validator("box")
    @classmethod
    def _box_valid(cls, v: list[float]) -> list[float]:
        _check_finite(v)
        if not (v[2] > v[0] and v[3] > v[1]):
            raise ValueError("box 必须满足 x2>x1 且 y2>y1（格式 [x1, y1, x2, y2]）")
        return v

    @field_validator("score")
    @classmethod
    def _score_finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("score 必须是有限数值")
        return v


class GroundTruthIn(BaseModel):
    id: int = Field(ge=0, description="真值框编号（同一张图内唯一）")
    box: list[float] = Field(min_length=4, max_length=4,
                             description="[x1, y1, x2, y2]，要求 x2>x1 且 y2>y1")

    @field_validator("box")
    @classmethod
    def _box_valid(cls, v: list[float]) -> list[float]:
        _check_finite(v)
        if not (v[2] > v[0] and v[3] > v[1]):
            raise ValueError("box 必须满足 x2>x1 且 y2>y1（格式 [x1, y1, x2, y2]）")
        return v


class ReviewRequest(BaseModel):
    predictions: list[PredictionIn] = Field(
        max_length=MAX_BOXES, description="NMS 抑制前的预测框，至多 60 个")
    ground_truths: list[GroundTruthIn] = Field(
        max_length=MAX_BOXES, description="真值框，至多 60 个")
    target_gt_ids: list[int] = Field(
        default_factory=list, max_length=MAX_BOXES,
        description="待找回（必须命中）的真值编号")
    iou_threshold: float = Field(ge=0.0, le=1.0,
                                 description="NMS 抑制阈值：IoU 严格大于它才抑制")
    exemptible_pairs: list[list[int]] = Field(
        default_factory=list, max_length=MAX_EXEMPT_PAIRS,
        description="可豁免的预测框编号对，至多 10 对；每对格式 [idA, idB]")

    @field_validator("exemptible_pairs")
    @classmethod
    def _pairs_valid(cls, v: list[list[int]]) -> list[list[int]]:
        for pair in v:
            if len(pair) != 2:
                raise ValueError("每个豁免对必须恰好包含 2 个预测框编号")
            if pair[0] == pair[1]:
                raise ValueError("豁免对的两个编号不能相同")
        return v

    @model_validator(mode="after")
    def _cross_check(self) -> "ReviewRequest":
        pred_ids = [p.id for p in self.predictions]
        if len(set(pred_ids)) != len(pred_ids):
            raise ValueError("预测框编号存在重复")
        gt_ids = [g.id for g in self.ground_truths]
        if len(set(gt_ids)) != len(gt_ids):
            raise ValueError("真值框编号存在重复")
        pred_id_set, gt_id_set = set(pred_ids), set(gt_ids)
        unknown_targets = sorted(set(self.target_gt_ids) - gt_id_set)
        if unknown_targets:
            raise ValueError(f"target_gt_ids 包含不存在的真值编号: {unknown_targets}")
        for pair in self.exemptible_pairs:
            unknown = [i for i in pair if i not in pred_id_set]
            if unknown:
                raise ValueError(f"豁免对 {pair} 包含不存在的预测框编号: {unknown}")
        return self


class SuppressionOut(BaseModel):
    prediction_id: int
    suppressed_by: int
    iou: float


class MatchOut(BaseModel):
    prediction_id: int
    gt_id: int
    iou: float


class MatchReportOut(BaseModel):
    kept_prediction_ids: list[int] = Field(description="NMS 后保留的预测框（按置信度降序）")
    suppressed: list[SuppressionOut] = Field(description="被抑制的预测框及其抑制者")
    matches: list[MatchOut] = Field(description="保留框与真值的配对明细")
    false_positive_prediction_ids: list[int] = Field(description="未配上真值的保留框（误报）")
    false_positive_count: int
    hit_gt_ids: list[int] = Field(description="被命中的真值编号")
    missed_gt_ids: list[int] = Field(description="未被命中的真值编号")
    recovered_target_gt_ids: list[int] = Field(description="指定待找回真值中已命中的部分")
    unrecovered_target_gt_ids: list[int] = Field(description="指定待找回真值中仍未命中的部分")


class ReviewResponse(BaseModel):
    status: str = Field(description="solved=找到最少豁免方案；infeasible=无解")
    min_exemption_count: int | None = Field(description="所需最少豁免对数（无解时为 null）")
    chosen_exempt_pairs: list[list[int]] = Field(description="选中的豁免框对")
    optimal_solution_count: int | None = Field(
        description="达到最少豁免数的可行方案个数（无解时为 null）")
    ignored_ineffective_pairs: list[list[int]] = Field(
        description="相互 IoU 未超过抑制阈值、豁免无意义而被忽略的候选对")
    explanation: str = Field(description="结论说明；无解时说明原因")
    baseline: MatchReportOut = Field(description="无豁免时的 NMS+匹配明细")
    after: MatchReportOut | None = Field(description="采用最少豁免后的明细（无解时为 null）")
    all_exempted: MatchReportOut | None = Field(
        description="豁免全部有效候选对后的明细（仅无解时给出，用于诊断）")


# ==================== 训练切片规划 ====================

class SliceGroundTruthIn(BaseModel):
    id: int = Field(ge=0, description="缺陷真值编号（同一张图内唯一）")
    box: list[int] = Field(
        min_length=4, max_length=4,
        description="[x1, y1, x2, y2]，整数像素，要求 x2>x1 且 y2>y1")

    @field_validator("id", mode="before")
    @classmethod
    def _id_valid(cls, v: object) -> object:
        return _check_int(v)

    @field_validator("box", mode="before")
    @classmethod
    def _box_valid(cls, v: object) -> object:
        return _check_int_box(v)


class SlicingRequest(BaseModel):
    image_width: int = Field(gt=0, description="原图宽度（整数像素）")
    image_height: int = Field(gt=0, description="原图高度（整数像素）")
    ground_truths: list[SliceGroundTruthIn] = Field(
        max_length=MAX_SLICE_GTS, description="缺陷真值框，至多 60 个")
    candidate_crops: list[list[int]] = Field(
        max_length=MAX_CANDIDATE_CROPS,
        description="候选裁框 [x1, y1, x2, y2]，整数像素，至多 12 个")

    @field_validator("image_width", "image_height", mode="before")
    @classmethod
    def _dim_valid(cls, v: object) -> object:
        return _check_int(v)

    @field_validator("candidate_crops", mode="before")
    @classmethod
    def _crops_valid(cls, v: object) -> object:
        if not isinstance(v, list):
            return v  # 交给 Pydantic 报类型错误
        for crop in v:
            if not isinstance(crop, list) or len(crop) != 4:
                raise ValueError("每个裁框必须恰好包含 4 个坐标 [x1, y1, x2, y2]")
            _check_int_box(crop)
        return v

    @model_validator(mode="after")
    def _cross_check(self) -> "SlicingRequest":
        w, h = self.image_width, self.image_height
        for i, crop in enumerate(self.candidate_crops):
            if not (0 <= crop[0] < crop[2] <= w and 0 <= crop[1] < crop[3] <= h):
                raise ValueError(f"候选裁框 {i} {crop} 超出原图范围（{w}×{h}）或不合法")
        gt_ids: list[int] = []
        for gt in self.ground_truths:
            gt_ids.append(gt.id)
            b = gt.box
            if not (0 <= b[0] < b[2] <= w and 0 <= b[1] < b[3] <= h):
                raise ValueError(f"真值框 {gt.id} {b} 超出原图范围（{w}×{h}）或不合法")
        if len(set(gt_ids)) != len(gt_ids):
            raise ValueError("缺陷真值编号存在重复")
        return self


class LabeledBoxOut(BaseModel):
    gt_id: int = Field(description="缺陷真值编号")
    box: list[int] = Field(description="[x1, y1, x2, y2]，以所属切片左上角为原点的整数像素坐标")


class SliceOut(BaseModel):
    crop_index: int = Field(description="选中裁框在请求 candidate_crops 中的序号（0 起）")
    crop: list[int] = Field(description="选中裁框在原图中的坐标 [x1, y1, x2, y2]")
    positive_boxes: list[LabeledBoxOut] = Field(
        description="完整正样本框（每个缺陷恰好出现在一个切片中），按缺陷编号升序")
    ignore_boxes: list[LabeledBoxOut] = Field(
        description="忽略框（被裁断/露出的缺陷区域），按缺陷编号升序；与正样本框无正面积重叠")


class SlicingResponse(BaseModel):
    status: str = Field(description="solved=找到最少切片方案；infeasible=无解（计划为空）")
    selected_crop_indices: list[int] = Field(description="入选裁框的候选序号（0 起，升序）")
    selected_crops: list[list[int]] = Field(description="入选裁框的原图坐标")
    plan: list[SliceOut] = Field(description="逐切片的正样本框与忽略框（无解时为空数组）")
    slice_count: int = Field(description="入选切片数量（无解时为 0）")
    optimal_solution_count: int | None = Field(
        description="同为最少切片数的可行方案个数（无解时为 null）")
    uncovered_gt_ids: list[int] = Field(
        description="无法被任一候选裁框完整包含的缺陷编号（仅无解时可能非空）")
    blocked_components: list[list[int]] = Field(
        description="相互重叠但无法共置于同一裁框的缺陷组（仅无解时可能非空）")
    explanation: str = Field(description="结论说明；无解时提示如何调整候选裁框")
