# Progress log

Dated "done / next" entries, newest last. One entry per session.

## 2026-10-10: session 0, bootstrap (Fable 5.1)

**Done.** Playbook steps 1 to 6 and 8: git and uv (Python 3.12), pyproject with ruff, strict
mypy and pytest, data guard extended with image and checkpoint suffixes and an `assets/`
allow-list (25 tests), git hooks installed and verified against a fake JPEG, Claude settings,
rules and the `/dryrun` and `/review-shapes` commands, `check.sh` green. CCT20 metadata
downloaded (3 MB) and inspected: `docs/data-inspection.md`. Class guideline and setup review saved
to `docs/course-guideline.md` and `docs/setup-review.md`. Public repo `codx999/who-goes-there`
created by the user (MIT, CCT20 citation in README); first CI run green (run 38096639477).

**Next.** Step 7 in a dedicated design session: `docs/plan.md` from HANDOFF.md, data-inspection.md,
course-guideline.md and setup-review.md; then CLAUDE.md, `docs/progress.json`. Then step 9
(Part B of `docs/review/playbook-reusable.md`) in its own session, and step 10.

Review: tier 0 for the docs; the guard script and tests are control-plane files and get their
first review with the step 9 machinery.
