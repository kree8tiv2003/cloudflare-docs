import numpy as np, pytest
from PIL import Image
from lumina_toolkit import get_feature, list_features
from lumina_toolkit.backends import BackendMissing, register_backend


def test_all_listed():
    assert len(list_features()) == 17


def test_upscale_size():
    assert get_feature("upscale").run(Image.new("RGB", (10, 8)), scale=3).size == (30, 24)


def test_remove_bg_makes_border_transparent():
    img = Image.new("RGB", (40, 40), (255, 255, 255))
    img.paste((200, 0, 0), (10, 10, 30, 30))
    out = get_feature("remove_bg").run(img)
    assert out.getpixel((0, 0))[3] == 0 and out.getpixel((20, 20))[3] == 255


def test_watermark_fill():
    img = Image.new("RGB", (30, 30), (100, 100, 100)); img.paste((255, 255, 255), (10, 10, 20, 20))
    mask = Image.new("L", (30, 30), 0); mask.paste(255, (10, 10, 20, 20))
    out = np.asarray(get_feature("remove_watermark").run(img, mask))
    assert abs(int(out[15, 15, 0]) - 100) < 5


def test_face_swap_needs_backend_then_works():
    f = get_feature("face_swap")
    with pytest.raises(BackendMissing):
        f.run(Image.new("RGB", (4, 4)), Image.new("RGB", (4, 4)))
    register_backend("face_swap", lambda s, t: t)
    assert f.run(Image.new("RGB", (4, 4)), Image.new("RGB", (5, 5))).size == (5, 5)
