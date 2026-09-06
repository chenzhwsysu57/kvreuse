import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from .attacker import Attacker
from .attack_judger import AttackJudger
from .defender import Defender
from .defend_judger import DefendJudger
from .executor import Executor
from .schemas import AttackRecord, DefenseJudgment, DefenseRecord, RoundRecord


class SelfPlayOrchestrator:
    def __init__(self, *, attacker: Attacker, attack_judger: AttackJudger,
                 defender: Defender, defend_judger: DefendJudger,
                 executor: Executor, max_rounds: int = 5,
                 max_attempts: int = 3,
                 attacks_per_round: int = 1,
                 defenses_per_round: int = 1,
                 attacker_workers: int = 4,
                 log_dir: str | None = None):
        self.attacker = attacker
        self.attack_judger = attack_judger
        self.defender = defender
        self.defend_judger = defend_judger
        self.executor = executor
        self.max_rounds = max_rounds
        self.max_attempts = max_attempts
        self.attacks_per_round = max(1, attacks_per_round)
        self.defenses_per_round = max(1, defenses_per_round)
        self.attacker_workers = max(1, attacker_workers)
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.attack_pool: list[AttackRecord] = []
        self.attack_history: list[str] = []
        self.defense_history: list[str] = []
        self.defense_pool: list[DefenseRecord] = []
        self._defense_baselines: dict[str, dict[str, object]] = {}

    def run(self, initial_remedy: str = "") -> str:
        remedy = initial_remedy
        for round_idx in range(1, self.max_rounds + 1):
            remedy_before = remedy
            print(f"[self-play] round {round_idx}/{self.max_rounds}; attacks={len(self.attack_pool)}", flush=True)
            attack_submissions = []
            accepted = None
            pending_attacks = []
            # Phase 1: collect all attacker proposals first.  Previously each
            # proposal was executed immediately, making attacks_per_round look
            # serial in the logs and allowing the first candidate to dominate.
            attack_history_snapshot = list(self.attack_history)
            defense_history_snapshot = list(self.defense_history)

            def generate_candidate(candidate_idx: int):
                errors = []
                for attempt in range(1, self.max_attempts + 1):
                    print(f"[self-play] attacker {candidate_idx + 1}/{self.attacks_per_round}; submission {attempt}/{self.max_attempts}", flush=True)
                    try:
                        case = self.attacker.propose_one(
                            attack_history=attack_history_snapshot,
                            defense_history=defense_history_snapshot,
                        )
                        return candidate_idx, (case, attempt, candidate_idx + 1), errors
                    except Exception as exc:
                        errors.append((attempt, repr(exc)))
                return candidate_idx, None, errors

            worker_count = min(self.attacker_workers, self.attacks_per_round)
            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                generated = list(pool.map(generate_candidate, range(self.attacks_per_round)))
            for candidate_idx, item, errors in generated:
                for attempt, error in errors:
                    attack_submissions.append({"candidate": candidate_idx + 1, "attempt": attempt, "error": error})
                    self._save_error(round_idx, "attacker_generation", {
                        "candidate": candidate_idx + 1, "attempt": attempt, "error": error,
                    })
                if item is not None:
                    pending_attacks.append(item)

            # Phase 2: batch independent full prefills where the backend can do
            # so, then judge each case separately.
            cases = [item[0] for item in pending_attacks]
            try:
                batched_executions = self.executor.execute_attack_batch(cases, remedy)
            except Exception as exc:
                batched_executions = [None] * len(pending_attacks)
                self._save_error(round_idx, "attacker_batch_executor", {
                    "cases": len(cases), "error": repr(exc),
                })
                print(f"[attacker batch executor error] {exc!r}", flush=True)
            for (case, attempt, candidate_idx), executions in zip(pending_attacks, batched_executions):
                try:
                    if executions is None:
                        raise RuntimeError("batch executor returned no result for this case")
                    judgment = self.attack_judger.judge(case, executions)
                    record = AttackRecord(case, executions, judgment, attempt)
                    attack_submissions.append(record.to_dict())
                    print(f"[attack judger] success={judgment.attack_success}: {judgment.summary}", flush=True)
                    self.attack_history.append(judgment.summary)
                    if judgment.attack_success:
                        if accepted is None:
                            accepted = record
                        self.attack_pool.append(record)
                except Exception as exc:
                    attack_submissions.append({"candidate": candidate_idx, "attempt": attempt, "error": repr(exc)})
                    self._save_error(round_idx, "attacker_executor", {
                        "candidate": candidate_idx, "attempt": attempt,
                        "error": repr(exc),
                    })
                    print(f"[attacker/executor error] {exc!r}", flush=True)

            defense_submissions = []
            if accepted is not None:
                for candidate_idx in range(self.defenses_per_round):
                    for attempt in range(1, self.max_attempts + 1):
                        print(f"[self-play] defender {candidate_idx + 1}/{self.defenses_per_round}; submission {attempt}/{self.max_attempts}", flush=True)
                        try:
                            proposal = self.defender.propose_one(
                                attack_history=self.attack_history,
                                defense_history=self.defense_history,
                            )
                            evaluations = {}
                            judgments = {}
                            for attack in self.attack_pool:
                                baseline = self._defense_baselines.get(attack.case.case_id)
                                executed = self.executor.execute_defense(
                                    attack.case, proposal.remedy,
                                    without_remedy=baseline,
                                )
                                if baseline is None:
                                    self._defense_baselines[attack.case.case_id] = executed["without_remedy"]
                                with_remedy = executed["with_remedy"]
                                without_remedy = executed["without_remedy"]
                                dj = self.defend_judger.judge(
                                    attack.case, proposal.remedy,
                                    {k: v for k, v in without_remedy.items() if k.startswith("reuse_")},
                                    with_remedy,
                                )
                                evaluations.update({
                                    f"{attack.case.case_id}:{key}": value
                                    for key, value in with_remedy.items()
                                })
                                judgments[attack.case.case_id] = dj.to_dict()
                                self.defense_history.append(dj.summary)
                            success = bool(judgments) and all(x["defense_success"] for x in judgments.values())
                            summary = " ; ".join(x["summary"] for x in judgments.values())
                            record = DefenseRecord(proposal, evaluations, DefenseJudgment(summary, success, details=judgments), attempt)
                            defense_submissions.append(record.to_dict())
                            print(f"[defend judger] success={success}: {summary}", flush=True)
                            if success:
                                remedy = proposal.remedy
                                self.defense_pool.append(record)
                                break
                        except Exception as exc:
                            defense_submissions.append({"candidate": candidate_idx + 1, "attempt": attempt, "error": repr(exc)})
                            self._save_error(round_idx, "defender_executor", {
                                "candidate": candidate_idx + 1, "attempt": attempt,
                                "error": repr(exc),
                            })
                            print(f"[defender/executor error] {exc!r}", flush=True)
                    if remedy != remedy_before:
                        break

            record = RoundRecord(round_idx, remedy_before, attack_submissions,
                                 accepted.to_dict() if accepted else None,
                                 defense_submissions, remedy)
            self._save_round(record)
        return remedy

    def _save_round(self, record: RoundRecord):
        if self.log_dir:
            path = self.log_dir / f"round_{record.round_idx:02d}.json"
            path.write_text(json.dumps(record.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def _save_error(self, round_idx: int, stage: str, details: dict):
        """Persist errors; terminal output alone is insufficient for long runs."""
        if not self.log_dir:
            return
        with (self.log_dir / "errors.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"round": round_idx, "stage": stage, **details},
                                    ensure_ascii=False) + "\n")
