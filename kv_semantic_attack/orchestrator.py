import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Sequence

from .attacker import Attacker
from .config import SelfPlayConfig
from .defender import Defender
from .executor import KVReuseExecutor
from .schemas import AttackCase, DefenseCandidate, RoundRecord, VerifiedAttack


class SelfPlayOrchestrator:
    def __init__(
        self,
        *,
        attacker: Attacker,
        defender: Defender,
        executor: KVReuseExecutor,
        config: SelfPlayConfig,
        log_dir: Optional[str] = None,
    ):
        self.attacker = attacker
        self.defender = defender
        self.executor = executor
        self.config = config
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        self.replay: List[VerifiedAttack] = []
        self.rounds: List[RoundRecord] = []

    def run(self, initial_correction: str = "") -> str:
        incumbent = initial_correction
        incumbent_score = 0.0
        no_improve_rounds = 0

        for round_idx in range(1, self.config.max_rounds + 1):
            history = self._select_history_for_attacker()

            # 1) attacker proposes candidate attacks
            attack_proposals = self.attacker.propose(
                correction=incumbent,
                history=history,
                n=self.config.attacks_per_round,
            )

            # 2) actual target system verifies attacks
            verified = [
                self.executor.verify_attack(
                    attack,
                    incumbent,
                    require_source_clean_success=self.config.require_source_clean_success,
                    require_target_clean_success=self.config.require_target_clean_success,
                )
                for attack in attack_proposals
            ]

            accepted = [x for x in verified if x.accepted]
            accepted.sort(key=lambda x: x.failure_strength, reverse=True)
            accepted = accepted[: self.config.accepted_attacks_per_round]

            self.replay.extend(accepted)
            self._trim_replay()

            # 如果这一轮没有产生真正 counterexample，仍允许 attacker 下一轮继续尝试；
            # 但 defender 没有新证据时不更新。
            if not accepted:
                record = RoundRecord(
                    round_idx=round_idx,
                    incumbent_before=incumbent,
                    attack_proposals=attack_proposals,
                    verified_attacks=verified,
                    defense_candidates=[],
                    incumbent_after=incumbent,
                    incumbent_score=incumbent_score,
                )
                self.rounds.append(record)
                self._save_round(record)
                no_improve_rounds += 1
                if no_improve_rounds >= self.config.patience:
                    break
                continue

            # 3) defender reads recent + representative historical failures
            defense_failures = self._select_history_for_defender(accepted)
            defense_candidates = self.defender.propose(
                incumbent=incumbent,
                failures=defense_failures,
                n=self.config.defenses_per_round,
                max_chars=self.config.max_correction_chars,
            )

            # 4) empirical scoring on replay buffer
            candidate_texts = [x.text for x in defense_candidates]
            if self.config.keep_incumbent:
                candidate_texts.append(incumbent)

            scored = []
            replay_attacks = [x.attack for x in self.replay]
            for text in candidate_texts:
                mean_score, worst_score = self.executor.score_correction(
                    text,
                    replay_attacks,
                )
                # 简单 robust objective；后续你可替换成 macro-average + worst-task
                robust_score = 0.7 * mean_score + 0.3 * worst_score
                scored.append((robust_score, mean_score, worst_score, text))

            scored.sort(reverse=True, key=lambda x: x[0])
            best_robust, best_mean, best_worst, best_text = scored[0]

            for c in defense_candidates:
                for robust, mean_score, worst_score, text in scored:
                    if c.text == text:
                        c.score = mean_score
                        c.worst_case_score = worst_score
                        break

            improved = best_robust > incumbent_score + self.config.min_improvement
            before = incumbent

            if improved:
                incumbent = best_text
                incumbent_score = best_robust
                no_improve_rounds = 0
            else:
                no_improve_rounds += 1

            record = RoundRecord(
                round_idx=round_idx,
                incumbent_before=before,
                attack_proposals=attack_proposals,
                verified_attacks=verified,
                defense_candidates=defense_candidates,
                incumbent_after=incumbent,
                incumbent_score=incumbent_score,
            )
            self.rounds.append(record)
            self._save_round(record)

            if no_improve_rounds >= self.config.patience:
                break

        return incumbent

    def _select_history_for_attacker(self) -> List[VerifiedAttack]:
        accepted = [x for x in self.replay if x.accepted]
        accepted.sort(key=lambda x: x.failure_strength, reverse=True)
        return accepted[: self.config.max_history_for_attacker]

    def _select_history_for_defender(
        self,
        recent: Sequence[VerifiedAttack],
    ) -> List[VerifiedAttack]:
        result = list(recent)
        recent_ids = {x.attack.attack_id for x in recent}

        historical = [
            x for x in self.replay
            if x.accepted and x.attack.attack_id not in recent_ids
        ]
        historical.sort(key=lambda x: x.failure_strength, reverse=True)

        room = max(0, self.config.max_history_for_defender - len(result))
        result.extend(historical[:room])
        return result[: self.config.max_history_for_defender]

    def _trim_replay(self):
        if len(self.replay) <= self.config.max_replay_cases:
            return
        self.replay.sort(key=lambda x: x.failure_strength, reverse=True)
        self.replay = self.replay[: self.config.max_replay_cases]

    def _save_round(self, record: RoundRecord):
        if not self.log_dir:
            return

        payload = {
            "round_idx": record.round_idx,
            "incumbent_before": record.incumbent_before,
            "attack_proposals": [x.to_dict() for x in record.attack_proposals],
            "verified_attacks": [x.to_dict() for x in record.verified_attacks],
            "defense_candidates": [x.to_dict() for x in record.defense_candidates],
            "incumbent_after": record.incumbent_after,
            "incumbent_score": record.incumbent_score,
        }
        path = self.log_dir / f"round_{record.round_idx:02d}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
