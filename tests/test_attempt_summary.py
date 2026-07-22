from prover.attempt_summary import (
    AttemptRecord,
    classify_local,
    replay_global_budget,
)


def test_classify_local_wall_counts_late_repair_proof_as_timeout():
    record = AttemptRecord(
        theorem="T",
        status="Proved",
        total_time=700.0,
        repair_time=250.0,
        actor_time=100.0,
        environment_time=350.0,
        num_total_nodes=10,
        num_searched_nodes=3,
    )

    assert classify_local(record, timeout=600.0, mode="wall") == "Timeout"
    assert classify_local(record, timeout=600.0, mode="effective") == "Proved"


def test_replay_global_budget_uses_selected_local_accounting_costs():
    records = [
        AttemptRecord("A", "Failed", 700.0, 250.0, 0.0, 0.0, 1, 1),
        AttemptRecord("B", "Proved", 100.0, 0.0, 0.0, 0.0, 1, 1),
    ]

    wall = replay_global_budget(records, timeout=600.0, mode="wall", global_budget=650.0)
    effective = replay_global_budget(records, timeout=600.0, mode="effective", global_budget=650.0)

    assert wall["attempted"] == 1
    assert wall["proved"] == 0
    assert effective["attempted"] == 2
    assert effective["proved"] == 1
