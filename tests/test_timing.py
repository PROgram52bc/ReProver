import pytest

from prover.timing import charged_time, should_stop_local


def test_wall_accounting_charges_all_elapsed_time():
    assert charged_time(total_time=700.0, repair_time=250.0, mode="wall") == 700.0


def test_effective_accounting_excludes_repair_time():
    assert charged_time(total_time=700.0, repair_time=250.0, mode="effective") == 450.0


def test_effective_accounting_never_goes_negative():
    assert charged_time(total_time=10.0, repair_time=25.0, mode="effective") == 0.0


def test_should_stop_local_uses_selected_accounting_mode():
    assert should_stop_local(total_time=700.0, repair_time=250.0, timeout=600.0, mode="wall")
    assert not should_stop_local(total_time=700.0, repair_time=250.0, timeout=600.0, mode="effective")


def test_invalid_timeout_accounting_mode_is_rejected():
    with pytest.raises(ValueError, match="timeout accounting"):
        charged_time(total_time=1.0, repair_time=0.0, mode="bad")
