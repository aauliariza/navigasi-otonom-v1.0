

def test_detect_head_mode_selects_nms_path():
    """D-6: default .pt inference uses the one-to-many head (NMS applies); end2end=True selects the NMS-free head."""
    import torch

    from models.common.model_utils import build_from_yaml

    m = build_from_yaml("yolo26n.yaml", "detect", nc=1).eval()
    assert m.end2end is False  # Ultralytics 8.4.194 default -> NMS with the configured IoU threshold
    with torch.no_grad():
        y = m(torch.zeros(1, 3, 64, 64))
        y = y[0] if isinstance(y, tuple) else y
        assert y.shape[1] == 4 + 1  # raw (B, 4 + nc, anchors) awaiting NMS
        m.end2end = True
        z = m(torch.zeros(1, 3, 64, 64))
        z = z[0] if isinstance(z, tuple) else z
        assert z.shape[-1] == 6  # (B, max_det, xyxy+conf+cls): already final, NMS-free
