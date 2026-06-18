# -*- coding: utf-8 -*-
"""Render the Section 6.4 diagrams (architecture, data-flow, sequence) as PNGs
using PyQt5/QPainter off-screen. No third-party dependencies beyond PyQt5
(already bundled with the application). Run:

    QT_QPA_PLATFORM=offscreen ./python/python.exe report_diagrams/generate_diagrams.py
"""

import os
import math

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import (
    QImage, QPainter, QColor, QFont, QPen, QBrush, QPolygonF, QFontDatabase,
    QPainterPath,
)
from PyQt5.QtCore import Qt, QRectF, QPointF

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
app = QApplication([])

# The offscreen QPA platform ships no fonts, so register system TTFs explicitly.
_SEGOE = [
    "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/segoeuii.ttf", "C:/Windows/Fonts/segoeuiz.ttf",
]
_ARIAL = [
    "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/ariali.ttf", "C:/Windows/Fonts/arialbi.ttf",
]
for _f in _SEGOE + _ARIAL:
    if os.path.exists(_f):
        QFontDatabase.addApplicationFont(_f)
_fams = set(QFontDatabase().families())
FAMILY = "Segoe UI" if "Segoe UI" in _fams else ("Arial" if "Arial" in _fams else
                                                 (sorted(_fams)[0] if _fams else "Sans Serif"))

# ---- palette ----
INK = QColor("#222730")
MUTE = QColor("#5b6b7b")
BG = QColor("white")
ARROW = QColor("#56616c")

PKG = {
    "ui":        QColor("#4C78A8"),
    "renderer":  QColor("#59A14F"),
    "objects":   QColor("#B07AA1"),
    "wind_data": QColor("#3A9CA8"),
    "models":    QColor("#E1812C"),
    "assets":    QColor("#7F8C99"),
    "flow":      QColor("#4C78A8"),
    "io":        QColor("#7F8C99"),
    "user":      QColor("#E1812C"),
}


def font(size, bold=False, italic=False):
    f = QFont(FAMILY)
    f.setPixelSize(size)
    f.setBold(bold)
    f.setItalic(italic)
    return f


SS = 3  # supersampling factor for high-quality (crisp) output


def new_canvas(w, h):
    img = QImage(w * SS, h * SS, QImage.Format_ARGB32)
    img.fill(BG)
    dpm = int(round(220 / 0.0254))  # ~220 DPI metadata so Word inserts at a sensible size
    img.setDotsPerMeterX(dpm)
    img.setDotsPerMeterY(dpm)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    p.setRenderHint(QPainter.TextAntialiasing)
    p.setRenderHint(QPainter.SmoothPixmapTransform)
    p.scale(SS, SS)  # draw in logical coordinates; output is SS× resolution
    return img, p


def light(color, alpha=28):
    c = QColor(color)
    c.setAlpha(alpha)
    return c


def text(p, rect, s, fnt, color=INK, align=Qt.AlignCenter, wrap=False):
    p.setFont(fnt)
    p.setPen(QPen(color))
    flow = int(Qt.TextWordWrap) if wrap else int(Qt.TextSingleLine)
    p.drawText(QRectF(*rect), int(align) | flow, s)


def chip(p, x, y, w, h, label, border, sub=None):
    p.setPen(QPen(border, 2))
    p.setBrush(QBrush(QColor("white")))
    p.drawRoundedRect(QRectF(x, y, w, h), 8, 8)
    if sub:
        text(p, (x, y + 6, w, h * 0.55), label, font(20, bold=True), INK)
        text(p, (x, y + h * 0.52, w, h * 0.45), sub, font(15), MUTE)
    else:
        text(p, (x, y, w, h), label, font(20, bold=True), INK)


def package(p, x, y, w, h, name, subtitle, chips, color):
    # outer
    p.setPen(QPen(color, 3))
    p.setBrush(QBrush(light(color, 22)))
    p.drawRoundedRect(QRectF(x, y, w, h), 14, 14)
    # header band
    hd = 48
    p.setBrush(QBrush(color))
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(QRectF(x, y, w, hd + 14), 14, 14)
    p.drawRect(QRectF(x, y + 14, w, hd))
    text(p, (x + 18, y, w * 0.40, hd + 6), name, font(23, bold=True), QColor("white"),
         Qt.AlignVCenter | Qt.AlignLeft)
    text(p, (x + w * 0.40, y, w * 0.60 - 26, hd + 6), subtitle, font(17),
         QColor("#f2f6fa"), Qt.AlignVCenter | Qt.AlignRight)
    # chips row
    n = len(chips)
    pad = 18
    gap = 16
    cy = y + hd + 26
    ch = h - hd - 50
    cw = (w - 2 * pad - gap * (n - 1)) / n
    for i, c in enumerate(chips):
        cx = x + pad + i * (cw + gap)
        chip(p, cx, cy, cw, ch, c, color)
    return  # caller uses box edges


def arrow(p, x1, y1, x2, y2, color=ARROW, width=3, dashed=False, head=13, label=None,
          label_dx=0, label_dy=-10):
    pen = QPen(color, width)
    pen.setCapStyle(Qt.RoundCap)
    if dashed:
        pen.setStyle(Qt.DashLine)
    p.setPen(pen)
    p.drawLine(QPointF(x1, y1), QPointF(x2, y2))
    ang = math.atan2(y2 - y1, x2 - x1)
    a1 = ang + math.radians(150)
    a2 = ang - math.radians(150)
    tip = QPointF(x2, y2)
    poly = QPolygonF([
        tip,
        QPointF(x2 + head * math.cos(a1), y2 + head * math.sin(a1)),
        QPointF(x2 + head * math.cos(a2), y2 + head * math.sin(a2)),
    ])
    p.setBrush(QBrush(color))
    p.setPen(Qt.NoPen)
    p.drawPolygon(poly)
    if label:
        mx, my = (x1 + x2) / 2 + label_dx, (y1 + y2) / 2 + label_dy
        text(p, (mx - 130, my - 14, 260, 28), label, font(15), MUTE)


def routed_path(p, pts, color=ARROW, width=2.5, dashed=True, radius=18, head=13):
    """Orthogonal poly-line through `pts` with rounded corners + an end arrowhead.
    Used to route dependency arrows around boxes instead of across them."""
    path = QPainterPath()
    path.moveTo(QPointF(*pts[0]))
    for i in range(1, len(pts) - 1):
        ax, ay = pts[i - 1]
        cx, cy = pts[i]
        bx, by = pts[i + 1]
        lin = math.hypot(cx - ax, cy - ay) or 1.0
        lout = math.hypot(bx - cx, by - cy) or 1.0
        r = min(radius, lin / 2, lout / 2)
        before = (cx - (cx - ax) / lin * r, cy - (cy - ay) / lin * r)
        after = (cx + (bx - cx) / lout * r, cy + (by - cy) / lout * r)
        path.lineTo(QPointF(*before))
        path.quadTo(QPointF(cx, cy), QPointF(*after))
    path.lineTo(QPointF(*pts[-1]))
    pen = QPen(color, width)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    if dashed:
        pen.setStyle(Qt.DashLine)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    p.drawPath(path)
    x1, y1 = pts[-2]
    x2, y2 = pts[-1]
    ang = math.atan2(y2 - y1, x2 - x1)
    poly = QPolygonF([
        QPointF(x2, y2),
        QPointF(x2 + head * math.cos(ang + math.radians(150)),
                y2 + head * math.sin(ang + math.radians(150))),
        QPointF(x2 + head * math.cos(ang - math.radians(150)),
                y2 + head * math.sin(ang - math.radians(150))),
    ])
    p.setBrush(QBrush(color))
    p.setPen(Qt.NoPen)
    p.drawPolygon(poly)


def title_block(p, w, title, sub):
    text(p, (0, 22, w, 36), title, font(28, bold=True), INK)
    text(p, (0, 62, w, 26), sub, font(17, italic=True), MUTE)


# =====================================================================
# DIAGRAM 1 — Architecture / Component
# =====================================================================
def diagram_architecture():
    W, H = 1640, 1180
    img, p = new_canvas(W, H)
    title_block(p, W, "Wind Visualization System — Component / Module Architecture",
                "Packages and key classes; arrows show the main direction of calls and data flow")

    x0, w = 90, W - 180
    # layer y/h
    ui_y = 120
    rn_y = 322
    mid_y = 524
    md_y = 726
    as_y = 928
    H1 = 150

    package(p, x0, ui_y, w, H1, "ui", "user interface & application loop",
            ["MainWindow", "ObjectLibraryPanel", "SimulationController"], PKG["ui"])
    package(p, x0, rn_y, w, H1, "renderer", "3D rendering, camera, picking",
            ["Scene", "Camera", "OpenGLWidget", "wind_colormap"], PKG["renderer"])

    half = (w - 60) / 2
    package(p, x0, mid_y, half, H1, "objects", "meshes & constraints",
            ["ObjectMesh"], PKG["objects"])
    package(p, x0 + half + 60, mid_y, half, H1, "wind_data", "wind field & OpenFOAM parsing",
            ["WindField", "openfoam_loader"], PKG["wind_data"])

    package(p, x0, md_y, w, H1, "models", "ML deformation (PyTorch + PyG)",
            ["DeformationModel", "MeshGraphNet", "config"], PKG["models"])
    package(p, x0, as_y, w, 130, "bundled assets", "shipped with the application",
            ["best_model.pth", "topology_edge_index.npy", "sample_openfoam_output/", "*.obj meshes"],
            PKG["assets"])

    cx = x0 + w / 2
    # ui -> renderer
    arrow(p, cx, ui_y + H1, cx, rn_y, width=3)
    # renderer -> objects / wind_data
    arrow(p, cx - 120, rn_y + H1, x0 + half / 2, mid_y, width=3)
    arrow(p, cx + 120, rn_y + H1, x0 + half + 60 + half / 2, mid_y, width=3)
    # objects/wind_data feed models (renderer & ui use models)
    arrow(p, x0 + half / 2, mid_y + H1, cx - 150, md_y, width=3)
    arrow(p, x0 + half + 60 + half / 2, mid_y + H1, cx + 150, md_y, width=3)
    # ui -> models : routed down the RIGHT gutter, around the boxes
    rg = x0 + w + 38
    routed_path(p, [(x0 + w, ui_y + H1 - 28), (rg, ui_y + H1 - 28),
                    (rg, md_y + 28), (x0 + w, md_y + 28)], width=2.5, dashed=True)
    # renderer -> models : routed down the LEFT gutter, around the boxes
    lg = x0 - 38
    routed_path(p, [(x0, rn_y + H1 - 28), (lg, rn_y + H1 - 28),
                    (lg, md_y + 28), (x0, md_y + 28)], width=2.5, dashed=True)
    # legend (bottom row, in the free space below the assets box)
    ly = as_y + 158
    p.setPen(QPen(ARROW, 3))
    p.drawLine(QPointF(x0 + 6, ly), QPointF(x0 + 54, ly))
    text(p, (x0 + 64, ly - 12, 240, 24), "calls / data flow", font(15), MUTE,
         Qt.AlignLeft | Qt.AlignVCenter)
    pen = QPen(ARROW, 3); pen.setStyle(Qt.DashLine); p.setPen(pen)
    p.drawLine(QPointF(x0 + 330, ly), QPointF(x0 + 378, ly))
    text(p, (x0 + 388, ly - 12, 560, 24), "also-uses dependency (e.g. ui & renderer → models)",
         font(15), MUTE, Qt.AlignLeft | Qt.AlignVCenter)
    # models -> assets
    arrow(p, cx, md_y + H1, cx, as_y, width=3, label="loads weights / topology", label_dy=-8)
    p.end()
    out = os.path.join(OUT_DIR, "diagram_1_architecture.png")
    img.save(out)
    return out


# =====================================================================
# DIAGRAM 2 — End-to-end data flow
# =====================================================================
def node(p, x, y, w, h, label, color, sub=None):
    p.setPen(QPen(color, 3))
    p.setBrush(QBrush(light(color, 22)))
    p.drawRoundedRect(QRectF(x, y, w, h), 12, 12)
    if sub:
        text(p, (x + 8, y + 10, w - 16, h * 0.5), label, font(20, bold=True), INK)
        text(p, (x + 8, y + h * 0.5, w - 16, h * 0.45), sub, font(15), MUTE)
    else:
        text(p, (x + 8, y, w - 16, h), label, font(20, bold=True), INK)
    return (x, y, w, h)


def diagram_dataflow():
    W, H = 1780, 800
    img, p = new_canvas(W, H)
    title_block(p, W, "Wind Visualization System — End-to-End Data Flow",
                "How data moves from OpenFOAM output to the rendered, deforming scene")

    bh = 96
    r1y = 175
    r2y = 470

    n1 = node(p, 60, r1y, 250, bh, "OpenFOAM case", PKG["io"], "postProcessing/, constant/")
    n2 = node(p, 400, r1y, 250, bh, "openfoam_loader", PKG["wind_data"], "parse .raw / boundary / STL")
    n3 = node(p, 740, r1y, 280, bh, "WindField", PKG["wind_data"], "5D array (c, z, y, x, t)")
    n4 = node(p, 1110, r1y, 290, bh, "SimulationController", PKG["ui"], "10 Hz tick loop")

    n5 = node(p, 120, r2y, 290, bh, "DeformationModel", PKG["models"], "MeshGraphNet inference")
    n6 = node(p, 470, r2y, 250, bh, "ObjectMesh", PKG["objects"], "deformed vertices")
    n7 = node(p, 780, r2y, 220, bh, "Scene", PKG["renderer"], "objects + wind + camera")
    n8 = node(p, 1060, r2y, 250, bh, "OpenGLWidget", PKG["renderer"], "draw frame")
    n9 = node(p, 1380, r2y, 200, bh, "Screen", PKG["io"], "rendered view")

    def right(n):
        return (n[0] + n[2], n[1] + n[3] / 2)

    def left(n):
        return (n[0], n[1] + n[3] / 2)

    def bottom(n):
        return (n[0] + n[2] / 2, n[1] + n[3])

    def top(n):
        return (n[0] + n[2] / 2, n[1])

    # row 1 chain
    arrow(p, *right(n1), *left(n2))
    arrow(p, *right(n2), *left(n3))
    arrow(p, *right(n3), *left(n4), label="query wind", label_dy=-12)
    # row1 -> row2 connector (SimulationController -> DeformationModel)
    sx, sy = bottom(n4)
    dx, dy = top(n5)
    arrow(p, sx, sy, sx, 360, width=3)
    arrow(p, sx, 360, dx, 360, width=3)
    arrow(p, dx, 360, dx, dy, width=3, label="predict()", label_dx=0, label_dy=-12)
    # row 2 chain
    arrow(p, *right(n5), *left(n6), label="next vertices", label_dy=-12)
    arrow(p, *right(n6), *left(n7))
    arrow(p, *right(n7), *left(n8), label="repaint", label_dy=-12)
    arrow(p, *right(n8), *left(n9))

    # user input branch
    uy = 660
    un = node(p, 690, uy, 320, 96, "User", PKG["user"], "drag-and-drop placement")
    arrow(p, 690 + 160, uy, 780 + 110, r2y + bh, width=3, dashed=True,
          label="add / move object", label_dy=-12)
    p.end()
    out = os.path.join(OUT_DIR, "diagram_2_dataflow.png")
    img.save(out)
    return out


# =====================================================================
# DIAGRAM 3 — Simulation-loop sequence
# =====================================================================
def diagram_sequence():
    W, H = 1780, 1160
    img, p = new_canvas(W, H)
    title_block(p, W, "Wind Visualization System — Simulation-Loop Sequence (one tick)",
                "SimulationController._simulation_step at 10 Hz (cfg.FPS, Δt = 0.1 s)")

    lifelines = [
        ("SimulationController", PKG["ui"]),
        ("WindField", PKG["wind_data"]),
        ("Scene", PKG["renderer"]),
        ("DeformationModel", PKG["models"]),
        ("ObjectMesh", PKG["objects"]),
        ("OpenGLWidget", PKG["renderer"]),
    ]
    n = len(lifelines)
    top_y = 120
    box_h = 64
    bottom_y = 1090
    xs = []
    margin = 70
    span = (W - 2 * margin) / n
    for i, (name, color) in enumerate(lifelines):
        cx = margin + span * (i + 0.5)
        xs.append(cx)
        bw = span - 36
        p.setPen(QPen(color, 3))
        p.setBrush(QBrush(light(color, 28)))
        p.drawRoundedRect(QRectF(cx - bw / 2, top_y, bw, box_h), 10, 10)
        text(p, (cx - bw / 2, top_y, bw, box_h), name, font(18, bold=True), INK)
        pen = QPen(QColor("#9aa6b2"), 2)
        pen.setStyle(Qt.DashLine)
        p.setPen(pen)
        p.drawLine(QPointF(cx, top_y + box_h), QPointF(cx, bottom_y))

    SC, WF, SCN, DM, OM, GL = xs

    def msg(y, a, b, label, dashed=False):
        arrow(p, a, y, b, y, width=2.6, dashed=dashed, head=12)
        # label above the line, aligned toward the source
        x1, x2 = sorted([a, b])
        text(p, (x1, y - 30, x2 - x1, 24), label, font(15), INK,
             Qt.AlignHCenter | Qt.AlignBottom)

    def selfmsg(y, x, label, side="right"):
        w = 58
        h = 32
        p.setPen(QPen(ARROW, 2.6))
        p.setBrush(Qt.NoBrush)
        sx = x + w if side == "right" else x - w
        p.drawLine(QPointF(x, y), QPointF(sx, y))
        p.drawLine(QPointF(sx, y), QPointF(sx, y + h))
        arrow(p, sx, y + h, x, y + h, width=2.6, head=11)
        if side == "right":
            text(p, (x + w + 16, y - 4, 600, h + 8), label, font(15), INK,
                 Qt.AlignLeft | Qt.AlignVCenter)
        else:
            text(p, (x - w - 16 - 600, y - 4, 600, h + 8), label, font(15), INK,
                 Qt.AlignRight | Qt.AlignVCenter)

    y = 238
    s = 70
    sj = 94  # advance after a self-message (taller)
    msg(y, SC, WF, "advance_time(1)"); y += s
    msg(y, SC, SCN, "get_wind_at_object(obj)"); y += s
    msg(y, SCN, WF, "get_velocity_at_position()"); y += s
    msg(y, WF, SCN, "wind velocity", dashed=True); y += s + 6
    msg(y, SC, DM, "predict(verts, wind, prev, rest_lengths)"); y += s + 6
    selfmsg(y, DM, "MeshGraphNet forward → re-pin → cap step → XPBD strain projection"); y += sj
    msg(y, DM, SC, "next vertices", dashed=True); y += s + 6
    selfmsg(y, SC, "_apply_constraints()  (pole edge / corners / base)"); y += sj
    msg(y, SC, OM, "update_vertices()  → recompute normals"); y += s
    msg(y, SC, GL, "simulation_updated  →  update()"); y += s
    selfmsg(y, GL, "paintGL()  —  redraw grid, wind, objects", side="left")

    p.end()
    out = os.path.join(OUT_DIR, "diagram_3_sequence.png")
    img.save(out)
    return out


if __name__ == "__main__":
    for fn in (diagram_architecture, diagram_dataflow, diagram_sequence):
        path = fn()
        print("wrote", os.path.relpath(path, OUT_DIR), os.path.getsize(path), "bytes")
