import numpy as np

from pipeline.ruled import find_ruled_lines


def ruled_page(n_rules=12, spacing=90, h=1200, w=800):
    img = np.full((h, w, 3), 245, np.uint8)
    for k in range(n_rules):
        img[100 + k * spacing:103 + k * spacing, 40:760] = (222, 222, 235)  # faint bluish rule, lighter than the ink threshold
    return img


def test_ruled_page_gives_one_box_per_inked_band():
    img = ruled_page()
    for dy in range(8, 32, 8):  # thin pen strokes above rule 2 and rule 5 (a solid block would swallow the background)
        img[190 - dy:190 - dy + 3, 200:600] = 20
        img[460 - dy:460 - dy + 3, 150:500] = 20
    boxes = find_ruled_lines(img)
    assert boxes is not None and len(boxes) == 2
    assert all(b[2] - b[0] >= 300 for b in boxes) and boxes[0][1] < boxes[1][1]


def test_plain_page_is_not_ruled():
    assert find_ruled_lines(np.full((1200, 800, 3), 245, np.uint8)) is None
