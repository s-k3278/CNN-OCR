"""Classical (OpenCV + NumPy) text-region detector. CPU only, no extra models.

Pipeline
--------
1. Edge mask      morphological gradient + Otsu threshold. An edge map has no
                  "text colour", so dark-on-light, light-on-dark, coloured
                  header bands and unevenly lit pages are all handled alike.
2. Line removal   long thin horizontal / vertical edge runs (table rules,
                  underlines, box borders) are subtracted so they cannot glue
                  text to the grid.
3. Characters     connected components of what is left.
4. Words          components are linked into text lines, then every line is
                  split at its own word gaps (threshold taken from that line's
                  gap pattern, scaled by glyph height). A 12 px footnote, a
                  60 px title and a widely spaced single word are all handled.
5. Clean-up       tiny marks (i-dots, accents) are absorbed into their word,
                  noise / frames / rules are dropped, overlaps are merged and
                  the boxes are put in reading order.

Boxes are ``(x, y, w, h)`` tuples in the coordinates of the image passed in.

Modes
-----
``word``      the image is one cropped word (IIIT-5K style) -> whole image.
``document``  always run the detector.
``auto``      small images that look like one word -> whole image, everything
              else -> detector.
"""
import cv2
import numpy as np

SMALL_H, SMALL_W = 160, 600  # IIIT-5K style word crops are at most this big


class TextDetector:
    def __init__(
        self,
        min_area=6,
        min_height=6,
        word_gap=0.38,
        link_gap=1.5,
        blob_gap=0.2,
        jump_ratio=2.0,
        min_word_gap=0.2,
        min_contrast=30,
        edge_factor=0.4,
        weak_floor=15,
        pad=0.12,
        mode="auto",
    ):
        self.min_area = min_area
        self.min_height = min_height
        self.word_gap = word_gap  # fallback gap (x line height) that separates words
        self.link_gap = link_gap  # max gap (x height) for two glyphs to share a line
        self.blob_gap = blob_gap  # word gap between two wide (fused-word) blobs
        self.jump_ratio = jump_ratio  # how much bigger a word gap is than a letter gap
        self.min_word_gap = min_word_gap  # smallest gap (x height) that can be a word gap
        self.min_contrast = min_contrast  # floor for the edge threshold
        self.pad = pad  # box padding as a fraction of box height
        self.edge_factor = edge_factor  # faint-edge threshold = this x Otsu
        self.weak_floor = weak_floor  # lowest allowed faint-edge threshold
        self.mode = mode

    # ------------------------------------------------------------------ API
    def detect(self, image, mode=None):
        gray = self._to_gray(image)
        h, w = gray.shape[:2]
        if h == 0 or w == 0:
            return []
        mode = mode or self.mode
        if mode not in {"auto", "word", "document"}:
            raise ValueError("mode must be 'auto', 'word' or 'document'")

        if mode == "word":
            return [(0, 0, w, h)]

        boxes = self._detect_words(gray)

        if mode == "auto" and h <= SMALL_H and w <= SMALL_W:
            if len(boxes) <= 1 or self._single_word_line(boxes):
                return [(0, 0, w, h)]
        return boxes

    # ------------------------------------------------------------ internals
    @staticmethod
    def _to_gray(image):
        if image.ndim == 2:
            gray = image
        elif image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return gray

    def _masks(self, gray):
        """Return (strong, weak) edge masks.

        ``strong`` (Otsu threshold) finds the words. ``weak`` (a lower
        threshold) is only used to adopt faint letters that sit right next to
        a detected word, so noise elsewhere cannot create false detections.
        """
        h, w = gray.shape
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        grad = cv2.morphologyEx(
            blur, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        )
        otsu, _ = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        hi = max(float(otsu), float(self.min_contrast))
        lo = max(self.edge_factor * float(otsu), float(self.weak_floor))
        strong = self._clean((grad >= hi), h, w)
        weak = self._clean((grad >= min(lo, hi)), h, w)
        return strong, weak

    @staticmethod
    def _clean(binary, h, w):
        mask = binary.astype(np.uint8) * 255
        # Remove long thin runs (table rules, underlines, borders).
        length = max(40, int(0.05 * max(h, w)))
        horiz = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
        )
        vert = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))
        )
        rules = cv2.dilate(horiz | vert, np.ones((3, 3), np.uint8))
        mask[rules > 0] = 0
        # Close tiny gaps in glyph outlines.
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    def _adopt_faint(self, groups, strong, weak, h, w):
        """Extend detected words with faint letters/strokes touching them."""
        extra = (weak > 0) & ~(cv2.dilate(strong, np.ones((5, 5), np.uint8)) > 0)
        n, _, stats, _ = cv2.connectedComponentsWithStats(extra.astype(np.uint8), connectivity=8)
        if n <= 1 or len(groups) == 0:
            return groups
        cx, cy, cw, ch, area = (stats[1:, i] for i in range(5))
        keep = (area >= 8) & (ch >= 4)
        cands = [
            (float(cx[i]), float(cy[i]), float(cx[i] + cw[i]), float(cy[i] + ch[i]), float(area[i]))
            for i in np.where(keep)[0]
        ]
        used = [False] * len(cands)
        for _ in range(3):
            changed = False
            for k, (x1, y1, x2, y2, a) in enumerate(cands):
                if used[k]:
                    continue
                gh = groups[:, 3] - groups[:, 1]
                ch_ = y2 - y1
                ov = np.minimum(groups[:, 3], y2) - np.maximum(groups[:, 1], y1)
                gap = np.maximum(np.maximum(groups[:, 0] - x2, x1 - groups[:, 2]), 0)
                ok = (
                    (ov >= 0.6 * np.minimum(gh, ch_))
                    & (gap <= 0.6 * gh)
                    & (ch_ >= 0.4 * gh)
                    & (ch_ <= 1.4 * gh)
                )
                if ok.any():
                    t = int(np.argmin(np.where(ok, gap, np.inf)))
                    groups[t, 0] = min(groups[t, 0], x1)
                    groups[t, 1] = min(groups[t, 1], y1)
                    groups[t, 2] = max(groups[t, 2], x2)
                    groups[t, 3] = max(groups[t, 3], y2)
                    groups[t, 4] += a
                    used[k] = True
                    changed = True
            if not changed:
                break
        return groups

    def _components(self, mask, h, w):
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            return np.zeros((0, 5), dtype=np.int32)
        s = stats[1:]
        x, y, bw, bh, area = s[:, 0], s[:, 1], s[:, 2], s[:, 3], s[:, 4]
        keep = (area >= self.min_area) & (bw >= 1) & (bh >= 2)
        # frames / pictures / page borders
        keep &= ~((bh > 0.5 * h) & (bw > 0.5 * w))
        keep &= ~(bh > 0.9 * h)
        keep &= ~(bw > 0.9 * w)
        comps = np.stack([x, y, bw, bh, area], axis=1)[keep].astype(np.int32)
        if len(comps) > 8000:  # photograph / texture: keep the biggest only
            comps = comps[np.argsort(-comps[:, 4])[:8000]]
        return comps

    def _group(self, comps):
        """Components -> text lines -> words.

        1. Link components that share a text line (vertical overlap) and are
           reasonably close.
        2. For every line, look at the gaps between neighbouring components.
           Letter gaps are small and word gaps are larger, so the threshold is
           taken from the largest jump in that line's own sorted gaps. A line
           with no clear jump (a single word, even a widely spaced one) is
           kept whole.
        """
        n = len(comps)
        x1 = comps[:, 0].astype(np.float32)
        y1 = comps[:, 1].astype(np.float32)
        hh = comps[:, 3].astype(np.float32)
        x2 = x1 + comps[:, 2]
        y2 = y1 + hh
        ink = comps[:, 4].astype(np.float32)

        parent = np.arange(n)

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(n - 1):
            j = np.arange(i + 1, n)
            link = self.link_gap * np.maximum(hh[i], hh[j])
            near = (x1[i] - x2[j] <= link) & (x1[j] - x2[i] <= link)
            if not near.any():
                continue
            j = j[near]
            overlap = np.minimum(y2[i], y2[j]) - np.maximum(y1[i], y1[j])
            for k in j[overlap >= 0.5 * np.minimum(hh[i], hh[j])]:
                ri, rk = find(i), find(k)
                if ri != rk:
                    parent[rk] = ri

        roots = np.array([find(i) for i in range(n)])
        groups = []

        def box(idx):
            return [x1[idx].min(), y1[idx].min(), x2[idx].max(), y2[idx].max(), float(ink[idx].sum())]

        for r in np.unique(roots):
            idx = np.where(roots == r)[0]
            if len(idx) == 1:
                groups.append(box(idx))
                continue
            idx = idx[np.argsort(x1[idx])]
            big = hh[idx] >= 0.4 * hh[idx].max()
            line_h = float(np.median(hh[idx][big]))
            # Thick glyphs only yield outline fragments, which under-states the
            # text height; the line's own vertical extent is a safer floor.
            line_h = max(line_h, 0.75 * float(y2[idx].max() - y1[idx].min()))
            reach = np.maximum.accumulate(x2[idx])
            gaps = np.clip(x1[idx[1:]] - reach[:-1], 0, None)
            ng = gaps / max(line_h, 1.0)
            thr, found_jump = self._gap_threshold(ng)
            widths = x2[idx] - x1[idx]
            start = 0
            for k, g in enumerate(ng):
                if found_jump:
                    split = g > thr
                else:
                    # No letter/word jump on this line. Two wide blobs are two
                    # words (their letters are fused); narrow glyphs far apart
                    # are one widely spaced word.
                    wide = 0.5 * (widths[k] + widths[k + 1]) >= 0.9 * line_h
                    split = g > (self.blob_gap if wide else self.word_gap)
                if split:
                    groups.append(box(idx[start:k + 1]))
                    start = k + 1
            groups.append(box(idx[start:]))
        return np.array(groups, dtype=np.float32)

    def _gap_threshold(self, gaps):
        """Word-gap threshold (in units of line height) from one line's gaps.

        Letter gaps are small and word gaps are larger. Sort the gaps and take
        the FIRST clear jump up to a "word-sized" gap; if there is no clear
        jump (e.g. a single word) fall back to ``word_gap``.
        """
        thr, found = self.word_gap, False
        if len(gaps) >= 2:
            g = np.sort(gaps)
            ratio = g[1:] / np.maximum(g[:-1], 0.06)
            ok = (ratio >= self.jump_ratio) & (g[1:] >= self.min_word_gap)
            if ok.any():
                k = int(np.argmax(ok))  # first eligible jump
                thr, found = float((g[k] + g[k + 1]) / 2), True
        return thr, found

    @staticmethod
    def _absorb_marks(groups):
        """Merge tiny marks (i-dots, accents, stray punctuation) into the word below."""
        if len(groups) < 2:
            return groups
        gw = groups[:, 2] - groups[:, 0]
        gh = groups[:, 3] - groups[:, 1]
        alive = np.ones(len(groups), dtype=bool)
        for i in np.argsort(gh):  # smallest first
            if not alive[i]:
                continue
            cx = (groups[i, 0] + groups[i, 2]) / 2
            cy = (groups[i, 1] + groups[i, 3]) / 2
            big = alive & (gh >= 2.5 * gh[i]) & (gw >= gw[i])
            big[i] = False
            if not big.any() or gw[i] > 0.4 * gh[big].max() or gh[i] > 0.4 * gh[big].max():
                continue
            cand = np.where(big)[0]
            bh = gh[cand]
            ok = (
                (cx >= groups[cand, 0] - 0.15 * bh)
                & (cx <= groups[cand, 2] + 0.15 * bh)
                & (cy >= groups[cand, 1] - 0.35 * bh)
                & (cy <= groups[cand, 3] + 0.05 * bh)
                & (gw[i] <= 0.4 * bh)
                & (gh[i] <= 0.4 * bh)
            )
            if ok.any():
                t = cand[np.argmax(ok)]
                groups[t, 0] = min(groups[t, 0], groups[i, 0])
                groups[t, 1] = min(groups[t, 1], groups[i, 1])
                groups[t, 2] = max(groups[t, 2], groups[i, 2])
                groups[t, 3] = max(groups[t, 3], groups[i, 3])
                groups[t, 4] += groups[i, 4]
                alive[i] = False
                gw[t] = groups[t, 2] - groups[t, 0]
                gh[t] = groups[t, 3] - groups[t, 1]
        return groups[alive]

    def _detect_words(self, gray):
        h, w = gray.shape
        strong, weak = self._masks(gray)
        comps = self._components(strong, h, w)
        if len(comps) == 0:
            return []
        groups = self._absorb_marks(self._group(comps))
        groups = self._adopt_faint(groups, strong, weak, h, w)

        out = []
        for gx1, gy1, gx2, gy2, ink in groups:
            bw, bh = gx2 - gx1, gy2 - gy1
            if bh < self.min_height or bw < 3 or ink < 20:
                continue
            if bw / max(bh, 1) > 80:  # a surviving rule, not a word
                continue
            if bh > 0.5 * h and bw > 0.5 * w:
                continue
            pad_x = max(2, int(round(self.pad * bh)))
            pad_y = max(2, int(round(self.pad * bh)))
            x1 = max(0, int(gx1) - pad_x)
            y1 = max(0, int(gy1) - pad_y)
            x2 = min(w, int(np.ceil(gx2)) + pad_x)
            y2 = min(h, int(np.ceil(gy2)) + pad_y)
            out.append((x1, y1, x2 - x1, y2 - y1))

        return self._reading_order(self._merge_overlaps(out))

    @staticmethod
    def _merge_overlaps(boxes, containment=0.6):
        """Merge boxes that mostly overlap (e.g. a word detected twice)."""
        boxes = [list(b) for b in boxes]
        changed = True
        while changed and len(boxes) > 1:
            changed = False
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    a, b = boxes[i], boxes[j]
                    ix = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
                    iy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
                    if ix <= 0 or iy <= 0:
                        continue
                    inter = ix * iy
                    if inter >= containment * min(a[2] * a[3], b[2] * b[3]):
                        x1, y1 = min(a[0], b[0]), min(a[1], b[1])
                        x2 = max(a[0] + a[2], b[0] + b[2])
                        y2 = max(a[1] + a[3], b[1] + b[3])
                        boxes[i] = [x1, y1, x2 - x1, y2 - y1]
                        del boxes[j]
                        changed = True
                        break
                if changed:
                    break
        return [tuple(int(v) for v in b) for b in boxes]

    @staticmethod
    def _reading_order(boxes):
        """Top-to-bottom lines, left-to-right inside a line."""
        if not boxes:
            return boxes
        boxes = sorted(boxes, key=lambda b: b[1] + b[3] / 2)
        lines, cur = [], [boxes[0]]
        cy = boxes[0][1] + boxes[0][3] / 2
        ch = boxes[0][3]
        for b in boxes[1:]:
            by = b[1] + b[3] / 2
            if abs(by - cy) <= 0.5 * max(ch, b[3]):
                cur.append(b)
                cy = float(np.mean([q[1] + q[3] / 2 for q in cur]))
                ch = max(ch, b[3])
            else:
                lines.append(cur)
                cur, cy, ch = [b], by, b[3]
        lines.append(cur)
        return [b for line in lines for b in sorted(line, key=lambda q: q[0])]

    @staticmethod
    def _single_word_line(boxes):
        """True when several boxes are really one widely spaced word on one line."""
        ys1 = min(b[1] for b in boxes)
        ys2 = max(b[1] + b[3] for b in boxes)
        uh = ys2 - ys1
        for b in boxes:
            ov = min(ys2, b[1] + b[3]) - max(ys1, b[1])
            if ov < 0.5 * b[3]:
                return False
        ordered = sorted(boxes, key=lambda b: b[0])
        gaps = [
            ordered[i + 1][0] - (ordered[i][0] + ordered[i][2])
            for i in range(len(ordered) - 1)
        ]
        return max(gaps) <= 0.8 * uh

    # kept so older code that called the helper keeps working
    @staticmethod
    def _union(boxes):
        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[0] + b[2] for b in boxes)
        y2 = max(b[1] + b[3] for b in boxes)
        return x1, y1, x2 - x1, y2 - y1