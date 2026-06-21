#!/usr/bin/env python
"""Generate installer/app.ico for the Wind Visualization System.

Pure-Python (numpy + zlib) so it works in the portable runtime without Pillow.
Renders a rounded teal->blue tile with three white "wind streak" arcs and packs
several sizes into a PNG-based .ico (supported by Windows Vista+ / 10 / 11).

Run:  python\python.exe installer\make_icon.py
"""
import os
import struct
import zlib

import numpy as np

SIZES = [256, 128, 64, 48, 32, 16]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.ico")

TOP = np.array([31, 182, 201], dtype=np.float64)    # teal  #1FB6C9
BOT = np.array([23, 71, 166], dtype=np.float64)     # blue  #1747A6


def _smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def render(n):
    """Return an (n, n, 4) uint8 RGBA array."""
    ys, xs = np.mgrid[0:n, 0:n].astype(np.float64)
    # normalized 0..1
    u = xs / (n - 1)
    v = ys / (n - 1)

    # vertical gradient background
    rgb = TOP[None, None, :] * (1 - v[..., None]) + BOT[None, None, :] * v[..., None]

    # white wind-streak arcs
    streaks = [
        (0.34, 0.060, 0.55, 0.030),   # (center_y, amplitude, alpha, half-thickness)
        (0.52, 0.075, 0.85, 0.040),
        (0.70, 0.060, 0.55, 0.030),
    ]
    aa = 1.5 / n  # anti-alias width in normalized units
    for cy, amp, alpha, hw in streaks:
        curve = cy + amp * np.sin(2 * np.pi * u + 0.6)
        dist = np.abs(v - curve)
        mask = (1.0 - _smoothstep(hw, hw + aa, dist)) * alpha
        rgb = rgb * (1 - mask[..., None]) + np.array([255, 255, 255])[None, None, :] * mask[..., None]

    # rounded-corner alpha mask
    r = 0.18 * n
    cx = np.minimum(xs, n - 1 - xs)
    cyv = np.minimum(ys, n - 1 - ys)
    inside_x = cx >= r
    inside_y = cyv >= r
    corner = np.sqrt(np.maximum(r - cx, 0) ** 2 + np.maximum(r - cyv, 0) ** 2)
    alpha = np.where(inside_x | inside_y, 1.0, 1.0 - _smoothstep(r - 1.0, r + 0.5, corner))
    alpha = np.clip(alpha, 0.0, 1.0)

    out = np.empty((n, n, 4), dtype=np.uint8)
    out[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    out[..., 3] = (alpha * 255).astype(np.uint8)
    return out


def png_bytes(rgba):
    """Encode an (h, w, 4) uint8 array as a PNG byte string."""
    h, w, _ = rgba.shape

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    # filter byte 0 prefixed per scanline
    raw = bytearray()
    for row in rgba:
        raw.append(0)
        raw.extend(row.tobytes())
    idat = zlib.compress(bytes(raw), 9)
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def main():
    images = [(n, png_bytes(render(n))) for n in SIZES]

    header = struct.pack("<HHH", 0, 1, len(images))  # reserved, type=icon, count
    entries = bytearray()
    payload = bytearray()
    offset = 6 + 16 * len(images)
    for n, data in images:
        wbyte = 0 if n >= 256 else n
        entries += struct.pack(
            "<BBBBHHII",
            wbyte, wbyte, 0, 0, 1, 32, len(data), offset
        )
        payload += data
        offset += len(data)

    with open(OUT, "wb") as f:
        f.write(header)
        f.write(entries)
        f.write(payload)
    print(f"Wrote {OUT} ({len(header) + len(entries) + len(payload)} bytes, sizes={SIZES})")


if __name__ == "__main__":
    main()
