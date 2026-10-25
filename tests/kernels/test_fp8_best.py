"""``fp8.best``: the fastest fp8 recipe a device runs, as SAM 3's ``fp8: auto`` resolves it."""

from kohakugbc_models.kernels import fp8


def test_fp8_auto_takes_the_fastest_recipe_the_card_runs(monkeypatch):
    for cap, want in (((12, 0), "mx"), ((10, 0), "mx"), ((8, 9), "row"), ((8, 6), None)):
        monkeypatch.setattr(fp8.torch.cuda, "get_device_capability", lambda d, cap=cap: cap)
        assert fp8.best("cuda:0") == want, cap
    assert fp8.best("cpu") is None
