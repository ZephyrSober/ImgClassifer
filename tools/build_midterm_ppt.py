from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.enum.text import PP_ALIGN, MSO_AUTO_SIZE
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "midterm_defense_shufflenet_cpu.pptx"
NOTES = ROOT / "ppt_image_notes.md"
ASSETS = ROOT / "ppt_assets"

DATA_ROOT = ROOT / "dataset/cleaned_runs/cleaned_petimages_20260523_201516_066875"
PREVIEW = ROOT / "dataset/previews/small_image_collage_20260523_201516_066875.jpg"
SUMMARY = ROOT / "dataset/splits/cleaned_petimages_20260523_201516_066875/split_test20260317_val20260318/summary.json"
METRICS = ROOT / "runs/20260523_201658_cats_dogs_shufflenet_v2_quick_cpu/metrics/final_metrics.json"


W, H = Inches(13.333), Inches(7.5)

BG = RGBColor(248, 249, 250)
INK = RGBColor(28, 35, 42)
MUTED = RGBColor(92, 103, 115)
TEAL = RGBColor(0, 128, 122)
GREEN = RGBColor(48, 140, 90)
GOLD = RGBColor(196, 142, 43)
RED = RGBColor(180, 75, 68)
LINE = RGBColor(216, 222, 228)
PANEL = RGBColor(255, 255, 255)


def emu(value: float):
    return Inches(value)


def add_text(slide, text, x, y, w, h, size=24, bold=False, color=INK, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(emu(x), emu(y), emu(w), emu(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.name = "Noto Sans CJK SC"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return box


def add_title(slide, title, subtitle=None):
    add_text(slide, title, 0.55, 0.34, 8.8, 0.5, size=25, bold=True)
    if subtitle:
        add_text(slide, subtitle, 0.57, 0.88, 9.0, 0.32, size=10, color=MUTED)
    line = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, emu(0.55), emu(1.20), emu(1.05), emu(0.04))
    line.fill.solid()
    line.fill.fore_color.rgb = TEAL
    line.line.fill.background()


def add_panel(slide, x, y, w, h, fill=PANEL, line=LINE, radius=False):
    shape_type = MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE if radius else MSO_AUTO_SHAPE_TYPE.RECTANGLE
    shp = slide.shapes.add_shape(shape_type, emu(x), emu(y), emu(w), emu(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    shp.line.color.rgb = line
    shp.line.width = Pt(0.8)
    return shp


def add_chip(slide, text, x, y, w, color=TEAL):
    shp = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, emu(x), emu(y), emu(w), emu(0.34))
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    shp.line.fill.background()
    tf = shp.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = text
    r.font.name = "Noto Sans CJK SC"
    r.font.size = Pt(10)
    r.font.bold = True
    r.font.color.rgb = RGBColor(255, 255, 255)
    return shp


def add_bullets(slide, items, x, y, w, h, size=16, color=INK):
    box = slide.shapes.add_textbox(emu(x), emu(y), emu(w), emu(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    for idx, item in enumerate(items):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.text = item
        p.level = 0
        p.font.name = "Noto Sans CJK SC"
        p.font.size = Pt(size)
        p.font.color.rgb = color
        p.space_after = Pt(5)
    return box


def add_footer(slide, page, hint):
    add_text(slide, hint, 0.55, 7.05, 8.8, 0.22, size=8.5, color=MUTED)
    add_text(slide, f"{page}/6", 12.35, 7.05, 0.5, 0.22, size=8.5, color=MUTED, align=PP_ALIGN.RIGHT)


def add_image_cover(slide, img_path, x, y, w, h):
    with Image.open(img_path) as im:
        im = ImageOps.exif_transpose(im)
        src_w, src_h = im.size
    box_ratio = w / h
    src_ratio = src_w / src_h
    if src_ratio >= box_ratio:
        height = h
        width = h * src_ratio
        left = x - (width - w) / 2
        top = y
    else:
        width = w
        height = w / src_ratio
        left = x
        top = y - (height - h) / 2
    pic = slide.shapes.add_picture(str(img_path), emu(left), emu(top), width=emu(width), height=emu(height))
    return pic


def build_sample_grid() -> Path:
    ASSETS.mkdir(exist_ok=True)
    files = [
        DATA_ROOT / "Cat/0.jpg",
        DATA_ROOT / "Dog/0.jpg",
        DATA_ROOT / "Cat/100.jpg",
        DATA_ROOT / "Dog/100.jpg",
        DATA_ROOT / "Cat/101.jpg",
        DATA_ROOT / "Dog/101.jpg",
    ]
    thumb_w, thumb_h = 300, 220
    label_h = 38
    cols, rows = 3, 2
    canvas = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), (248, 249, 250))
    draw = ImageDraw.Draw(canvas)
    for i, path in enumerate(files):
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            fitted = ImageOps.fit(im, (thumb_w, thumb_h), method=Image.Resampling.LANCZOS)
        col, row = i % cols, i // cols
        x, y = col * thumb_w, row * (thumb_h + label_h)
        canvas.paste(fitted, (x, y))
        draw.rectangle((x, y + thumb_h, x + thumb_w, y + thumb_h + label_h), fill=(235, 239, 242))
        draw.text((x + 12, y + thumb_h + 10), path.parent.name, fill=(28, 35, 42))
    out = ASSETS / "cat_dog_sample_grid.jpg"
    canvas.save(out, quality=92)
    return out


def build_ppt():
    with SUMMARY.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    with METRICS.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    sample_grid = build_sample_grid()

    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H
    blank = prs.slide_layouts[6]

    # Slide 1
    slide = prs.slides.add_slide(blank)
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = BG
    add_chip(slide, "中期检查答辩 5分钟版", 0.65, 0.50, 1.85, TEAL)
    add_text(slide, "基于卷积神经网络的图片分类设计与开发", 0.65, 1.18, 6.9, 0.92, 31, True)
    add_text(slide, "MindSpore + Cats vs Dogs + ShuffleNetV2", 0.68, 2.18, 5.2, 0.35, 17, False, MUTED)
    add_panel(slide, 0.68, 3.05, 5.15, 2.0)
    add_bullets(
        slide,
        [
            "目标：完成猫狗二分类训练闭环",
            "当前进展：数据清理、划分、训练框架、ShuffleNetV2 初步训练已完成",
            "答辩重点：工程闭环与初步实验结果",
        ],
        0.95,
        3.38,
        4.55,
        1.25,
        15,
    )
    add_image_cover(slide, sample_grid, 7.05, 0.70, 5.65, 5.70)
    add_footer(slide, 1, "讲述重点：项目对象、当前已完成内容、答辩主线。")

    # Slide 2
    slide = prs.slides.add_slide(blank)
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = BG
    add_title(slide, "技术路线", "从原始图片到可复现实验结果")
    steps = ["原始数据", "数据清理", "数据划分", "manifest.csv", "Dataloader", "ShuffleNetV2", "验证/测试"]
    x0, y0 = 0.62, 2.25
    for i, step in enumerate(steps):
        x = x0 + i * 1.78
        w = 1.25 if i != 3 else 1.38
        add_panel(slide, x, y0, w, 0.78, fill=RGBColor(250, 252, 252), line=TEAL)
        add_text(slide, step, x + 0.08, y0 + 0.24, w - 0.16, 0.18, 10.5, True, INK, PP_ALIGN.CENTER)
        if i < len(steps) - 1:
            add_text(slide, "→", x + w + 0.06, y0 + 0.25, 0.35, 0.25, 18, True, TEAL, PP_ALIGN.CENTER)
    add_panel(slide, 1.0, 4.05, 3.3, 1.18)
    add_text(slide, "Manifest-first", 1.22, 4.23, 2.8, 0.28, 18, True, TEAL)
    add_text(slide, "训练阶段只读取 manifest 中固定的路径、标签和 split，避免临时扫目录造成划分混乱。", 1.22, 4.60, 2.72, 0.42, 11.5, False, INK)
    add_panel(slide, 5.0, 4.05, 3.3, 1.18)
    add_text(slide, "配置驱动", 5.22, 4.23, 2.8, 0.28, 18, True, GOLD)
    add_text(slide, "数据路径、模型、训练参数和运行环境都由 YAML 控制，便于切换实验。", 5.22, 4.60, 2.72, 0.42, 11.5, False, INK)
    add_panel(slide, 9.0, 4.05, 3.3, 1.18)
    add_text(slide, "闭环验证", 9.22, 4.23, 2.8, 0.28, 18, True, GREEN)
    add_text(slide, "已完成 train / val / test / checkpoint / metrics 输出流程。", 9.22, 4.60, 2.72, 0.42, 11.5, False, INK)
    add_footer(slide, 2, "图片建议：本页已经用流程图替代图片，无需额外放图。")

    # Slide 3
    slide = prs.slides.add_slide(blank)
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = BG
    add_title(slide, "数据清理与划分结果", "清理后数据量稳定，类别与 split 分布基本均衡")
    add_panel(slide, 0.65, 1.55, 5.1, 4.72)
    rows = [
        ("原始有效图片", "24,959"),
        ("移除小图", "514"),
        ("清理后图片", "24,445"),
        ("训练集", f"{summary['split_counts']['train']:,}"),
        ("验证集", f"{summary['split_counts']['val']:,}"),
        ("测试集", f"{summary['split_counts']['test']:,}"),
        ("转换 RGB", "51"),
    ]
    add_text(slide, "关键统计", 0.95, 1.82, 2.3, 0.28, 17, True, TEAL)
    for i, (k, v) in enumerate(rows):
        y = 2.22 + i * 0.45
        add_text(slide, k, 1.00, y, 2.6, 0.22, 12.5, False, MUTED)
        add_text(slide, v, 4.25, y, 0.95, 0.22, 13.5, True, INK, PP_ALIGN.RIGHT)
    add_panel(slide, 6.20, 1.55, 6.45, 4.72)
    add_image_cover(slide, PREVIEW, 6.45, 1.88, 5.95, 3.75)
    add_text(slide, "右图：小图样例预览，用于说明为何需要清理低质量图片。", 6.55, 5.78, 5.70, 0.26, 10.5, False, MUTED)
    add_footer(slide, 3, "图片建议：右侧放数据清理预览图或几张被移除的小图。")

    # Slide 4
    slide = prs.slides.add_slide(blank)
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = BG
    add_title(slide, "训练框架与 ShuffleNetV2", "先保证完整训练闭环，再逐步提升模型效果")
    blocks = [
        ("数据加载", ["读取 manifest.csv", "校验字段/路径/标签", "训练增强：裁剪、翻转、归一化"], TEAL),
        ("模型构建", ["Linear / SimpleCNN / ShuffleNetV2", "当前重点：ShuffleNetV2", "轻量化 CNN，适合后续部署"], GOLD),
        ("训练引擎", ["train_one_epoch", "validate / test", "保存 best checkpoint 与 metrics"], GREEN),
    ]
    for i, (title, bullets, color) in enumerate(blocks):
        x = 0.70 + i * 4.15
        add_panel(slide, x, 1.65, 3.65, 4.55)
        add_chip(slide, title, x + 0.28, 1.93, 1.25, color)
        add_bullets(slide, bullets, x + 0.35, 2.58, 2.95, 1.55, 13.2)
    add_text(slide, "配置文件切换实验：smoke / quick / baseline", 0.92, 6.45, 8.2, 0.30, 13.5, True, INK)
    add_text(slide, "smoke 用于快速检查链路；quick 用于完整数据 1 epoch；baseline 用于正式多轮训练。", 0.92, 6.80, 9.2, 0.25, 10.5, False, MUTED)
    add_footer(slide, 4, "图片建议：可替换为代码结构截图或 ShuffleNetV2 结构示意图。")

    # Slide 5
    slide = prs.slides.add_slide(blank)
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = BG
    add_title(slide, "ShuffleNetV2 初步训练结果", "CPU 上完成完整 cleaned 数据集 1 epoch 验证")
    add_panel(slide, 0.75, 1.55, 5.55, 4.8)
    add_text(slide, "实验设置", 1.05, 1.88, 2.4, 0.28, 17, True, TEAL)
    setting_rows = [
        ("模型", "ShuffleNetV2 x1.0"),
        ("设备", "CPU"),
        ("训练轮数", "1 epoch"),
        ("训练 step", "611"),
        ("验证 step", "77"),
        ("测试 step", "77"),
    ]
    for i, (k, v) in enumerate(setting_rows):
        y = 2.34 + i * 0.48
        add_text(slide, k, 1.08, y, 1.7, 0.22, 12.5, False, MUTED)
        add_text(slide, v, 3.2, y, 2.35, 0.22, 13, True, INK)

    add_panel(slide, 6.75, 1.55, 5.75, 4.8)
    add_text(slide, "指标", 7.05, 1.88, 1.4, 0.28, 17, True, GREEN)
    vals = [
        ("Train Loss", 0.646301, 1.0, TEAL),
        ("Val Acc", metrics["best_accuracy"], 1.0, GOLD),
        ("Test Acc", metrics["test_accuracy"], 1.0, GREEN),
    ]
    for i, (label, value, maxv, color) in enumerate(vals):
        y = 2.45 + i * 0.78
        add_text(slide, label, 7.08, y, 1.45, 0.22, 12.5, False, MUTED)
        bg = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, emu(8.65), emu(y + 0.03), emu(2.25), emu(0.18))
        bg.fill.solid()
        bg.fill.fore_color.rgb = RGBColor(229, 234, 238)
        bg.line.fill.background()
        bar = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, emu(8.65), emu(y + 0.03), emu(2.25 * min(value / maxv, 1.0)), emu(0.18))
        bar.fill.solid()
        bar.fill.fore_color.rgb = color
        bar.line.fill.background()
        display = f"{value:.4f}" if label == "Train Loss" else f"{value * 100:.2f}%"
        add_text(slide, display, 11.10, y - 0.02, 0.85, 0.24, 12.5, True, INK, PP_ALIGN.RIGHT)
    add_text(slide, "结论：ShuffleNetV2 训练闭环已经跑通；当前 1 epoch 结果主要用于中期验证，后续通过多轮训练和调参继续提升。", 7.08, 5.25, 4.75, 0.55, 11.8, False, INK)
    add_footer(slide, 5, "图片建议：可替换为训练曲线截图；目前用指标条形图展示。")

    # Slide 6
    slide = prs.slides.add_slide(blank)
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = BG
    add_title(slide, "问题与下一步计划", "当前已达到中期目标，后续重点转向效果优化与展示完善")
    add_panel(slide, 0.75, 1.58, 5.75, 4.85)
    add_text(slide, "当前问题", 1.05, 1.90, 2.0, 0.28, 17, True, RED)
    add_bullets(
        slide,
        [
            "RTX 5070 与当前 MindSpore GPU 运行库兼容性有限，本阶段采用 CPU 训练",
            "目前只完成 1 epoch，模型还未充分收敛",
            "结果展示还需要补充训练曲线和混淆矩阵",
        ],
        1.08,
        2.42,
        4.75,
        1.9,
        13.2,
    )
    add_panel(slide, 6.85, 1.58, 5.65, 4.85)
    add_text(slide, "下一步计划", 7.15, 1.90, 2.0, 0.28, 17, True, TEAL)
    add_bullets(
        slide,
        [
            "增加训练轮数，记录完整 loss/accuracy 曲线",
            "调整学习率、batch size、优化器和数据增强策略",
            "增加混淆矩阵与单图预测演示",
            "整理最终报告和结题答辩材料",
        ],
        7.18,
        2.42,
        4.75,
        2.1,
        13.2,
    )
    add_panel(slide, 2.15, 6.18, 9.1, 0.62, fill=RGBColor(238, 248, 246), line=TEAL)
    add_text(slide, "中期结论：数据处理、训练框架和 ShuffleNetV2 初步训练验证已经完成，项目可以进入效果优化阶段。", 2.45, 6.38, 8.5, 0.22, 12.5, True, INK, PP_ALIGN.CENTER)
    add_footer(slide, 6, "图片建议：本页不建议放图，保持结论和计划清晰。")

    prs.save(OUT)

    NOTES.write_text(
        "\n".join(
            [
                "# PPT 图片放置说明",
                "",
                "PPT 文件：`midterm_defense_shufflenet_cpu.pptx`",
                "",
                "## 第 1 页 项目简介",
                "- 已放图片：猫狗样例拼图 `ppt_assets/cat_dog_sample_grid.jpg`。",
                "- 可替换图片：从 cleaned 数据集中任选 3 张猫、3 张狗，做成样例拼图。",
                "",
                "## 第 2 页 技术路线",
                "- 已用 PPT 形状画流程图，不需要额外图片。",
                "- 可替换图片：如果老师更喜欢代码证据，可放项目目录结构截图。",
                "",
                "## 第 3 页 数据清理与划分",
                "- 已放图片：`dataset/previews/small_image_collage_20260523_201516_066875.jpg`。",
                "- 可替换图片：数据清理预览图、小图样例图、或清理前后对比图。",
                "",
                "## 第 4 页 训练框架与 ShuffleNetV2",
                "- 当前未放外部图片，使用模块职责卡片。",
                "- 可替换图片：ShuffleNetV2 网络结构示意图，或 `code/train/src/` 目录结构截图。",
                "",
                "## 第 5 页 实验结果",
                "- 当前用指标条形图展示结果。",
                "- 可替换图片：后续多 epoch 训练后，把 loss/accuracy 曲线图放在右侧。",
                "",
                "## 第 6 页 问题与计划",
                "- 不建议放图片，保留问题和计划即可。",
                "",
                "## 5 分钟讲述节奏",
                "- 第 1 页：30 秒。",
                "- 第 2 页：45 秒。",
                "- 第 3 页：50 秒。",
                "- 第 4 页：50 秒。",
                "- 第 5 页：60 秒。",
                "- 第 6 页：45 秒。",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    build_ppt()
