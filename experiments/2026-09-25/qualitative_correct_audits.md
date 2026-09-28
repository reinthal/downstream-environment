# Qualitative analysis: games where the auditor caught the Parsnip

Data: the 107 (game, arm) pairs from the 2026-09-25 grade-annotated auditor run
(local, 1 audit per condition) in which at least one condition's auditor named
the Parsnip. Each case was judged by an LLM subagent against a fixed rubric
(slip-up type from public text, whether the auditor's stated reasoning was
evidence-based / generic / wrong, round-1-2 dynamics); 105 returned valid
judgments. Raw per-game judgments: scratchpad `qual/compiled.json` (session
artifact, summarized here).

## 1. How many wins came from Parsnip slip-ups?

**77/105 (73%)** of correctly-audited games contain a detectable public
slip-up by the Parsnip:

| slip-up type | n | share | of which auditor still only lucky |
|---|---|---|---|
| behavioral_tell (stalling, deferral, zero-content agreement) | 40 | 38% | 6 |
| strategic_backfire (counter-accusation / lone defense of the scapegoat / criterion that described themselves) | 24 | 23% | 2 |
| self_contradiction (flip-flops, timeline errors) | 8 | 8% | 1 |
| think_leak (private strategy pasted into public text, incl. literal "I am the Parsnip") | 5 | 5% | 0 |
| **none** (publicly indistinguishable from Carrots) | 28 | 27% | 20 |

The modal catch is not a lie being detected — it is **passivity**: the Parsnip
contributes nothing concrete, defers, stalls the vote, and the group (and
auditor) reads sustained content-free agreement as the tell. The second most
common is the Parsnip's own **gambit backfiring** — a counter-accusation,
defending the group's scapegoat (protecting a Carrot the group was about to
eliminate, which would have won the game — several Parsnips misplay their win
condition), or proposing an elimination criterion ("vote out the vaguest")
that described their own play.

## 2. How many were pure luck?

**29/105 (28%)** — the auditor's reasoning across all correct conditions was
generic vibes or demonstrably wrong (misquoted transcript, case built against
a different player, or anchored on the random probe's noise grades) and the
correct name appears essentially unexplained. Luck concentrates exactly where
it should: in 20 of the 28 no-slip games the win was lucky, versus 9 of 77
slip-up games. Several lucky wins are visibly anchored on **random-condition
noise grades** that happened to peak on the Parsnip; a recurring failure shape
is reasoning that argues for player X and then answers player Y.

Auditor reasoning quality overall: evidence_based 76 / generic_heuristic 18 /
wrong_reason 11.

## 3. Are the two "dead rounds" actually dead?

No. **90/105 judgments describe substantive round-1-2 dynamics**; only 3 were
boilerplate-only. Recurring patterns:

- **The case is usually complete before round 3.** The typical arc: round 1
  uniform introductions, round 2 someone demands specifics, the Parsnip's
  evasive answer becomes the evidence, round 3 merely executes the vote.
- **Round-1 boilerplate is weaponized**, especially in the steered arm: the
  uniformity itself gets flagged ("the Parsnip is hiding in the safe script"),
  and whoever *defends* the uniformity — often the Parsnip — becomes the
  suspect.
- **Seeding and backfires**: criteria proposed in rounds 1-2 ("watch who
  avoids taking a risk") become traps their proposers or the Parsnip walk into
  in round 3; several Parsnips seeded a narrative early ("too-perfect
  mirroring") and successfully rode it to eliminate a Carrot.
- **Harness artifacts shape many games**: 26/105 cases mention think-leaks or
  truncation glitches, and in at least 14 the leaker is an innocent **Carrot**
  whose leaked `<thinking>`/role-claim hijacks the whole game into a wrong
  lynch (the Parsnip then just bandwagons — several such games are exactly the
  no-slip lucky-auditor cases).

## Arm differences

Steered Parsnips show more behavioral tells (25/57 = 44% vs 15/48 = 31% base)
and more self-contradictions (5 vs 3), consistent with the arm-pooled
accuracy gap: steering leaks into observable behavior — stalling, incoherent
lone-dissents, defending scapegoats against their own win condition — rather
than making detector read-outs more useful.

## Caveats

Single LLM judge per game (no adversarial verification); rubric labels on
~2.5k-char reasoning excerpts; the 107 games oversample auditor-friendly
games by construction (conditioning on a correct audit). Counts should be
read as characterization, not precise measurement.
