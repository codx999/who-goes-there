---
description: Review this session's commits: code-reviewer on the full range, plus risk-reviewer on high-stakes files
disable-model-invocation: true
---
Run the session review (CLAUDE.md, session protocol step 5; `docs/setup-playbook.md`, "Code review
playbook"). Do not edit anything in this command. Only the user starts it; never run it on your own.

Arguments (all optional): a range `<a>..<b>` to review only part of the branch (fix rounds) or
more than the branch (commits already merged to main, e.g. `9788ae4..HEAD` for the first review),
and item ids (e.g. `3.02 3.03`) naming the `docs/progress.json` items the range implements. The
implementing session knows the items; pass them when you type the command in that session. When a
range is given, use it everywhere below in place of `<base>..HEAD`.

1. If `git status --porcelain` shows changes, stop and ask to commit first: the reviewers see
   commits only.
2. Compute the range unless one was given: run `git merge-base main HEAD` and use the commit id
   it prints, written out literally, as `<base>` from here on (shell variables do not survive
   between Bash calls). If the current branch is `main`, ask for the range instead. If `HEAD`
   equals `<base>`, report that there is nothing to review and stop. After a mid-session merge of
   `main`, the range is still `merge-base..HEAD`; say so in the report if part of it was reviewed
   before and the user may narrow it.
3. Run `bash scripts/risk_files.sh <base>..HEAD` and keep its output (one path per line; empty
   means no high-stakes files changed). If it exits non-zero, stop and report the error.
4. Pick the scenarios: open `docs/review/invariants.md` and select every row whose "where" column
   matches a printed path (prefix or glob: `trailcam/data/**` matches any file under it,
   `*.ipynb` any notebook, `trailcam/train/train.py` that file) or a given item id. Also run
   `git diff --name-only <base>..HEAD -- docs/ CLAUDE.md .claude/rules/` and select rows whose
   "source" document is in that list: report those as "spec changed: rows ..." so the session
   re-reads them before merging. Known gaps to skip come from the user or the implementing session
   as a short list of ids; never paste progress-log narrative to a reviewer.
5. Report before spawning anything, in this order:
   - run items first: if a given item's steps are `manual:` (Kaggle training runs, the cache
     build, the final test scoring, the Space deployment), its unit of review is the ledger row
     the code writes (invariant R40), not a diff. Name the ledger rows the progress-log entry must
     quote; the implementing session checks them against R40 and R25 (arm rows with the same seed
     share split hash and config apart from arm) and the user reads them. The diff that built the
     ledger code is reviewed below like any other.
   - no code changed (docs, slides, rehearsal) and no run item: tier 0; report any "spec changed"
     rows and stop.
   - step 3 printed nothing: tier 1; tell the user to run
     `/code-review xhigh --max-findings all <base>..HEAD` instead, and stop.
   - step 3 printed files: tier 2, continue.
   - the given items include 3.03 (the split code) or 9.01 (the final scoring script), or the
     printed files include `trailcam/data/split.py`, `trailcam/evaluate/score_test.py`,
     `tests/test_split.py` or `tests/test_score_test.py` (their tests: a skipped leakage test is a
     tier-3 change): tier 3;
     after the tier-2 reports, remind the user to run `/code-review ultra` with time for one fix
     round before the deadline.
6. In one message, so the two run in parallel, spawn. Every reviewer prompt starts with: "Read
   `.claude/agents/<name>.md` first and follow it as your instructions; the copy this session
   registered at start may be stale." (Agent files load at session start; a stale-but-present
   registry runs the old prompt without any error.)
   - `code-reviewer` with `model: opus` and `effort: xhigh`, prompt: that line, then "Review the
     range `<base>..HEAD`. Scenarios to trace: <selected rows, id and scenario text>. Known gaps
     to skip: <list>."
   - `risk-reviewer` with `model: fable` and `effort: high`, prompt: that line, then "Review the
     changes in `<base>..HEAD` to these files: <list from step 3>. Scenarios to trace: <selected
     rows>." If Fable 5.1 implemented the item (the progress-log heading or the commit trailer
     says so), spawn a second `code-reviewer` (`model: opus`, `effort: xhigh`) on those files
     instead, so the second opinion comes from a different model than the one that wrote the
     code.

   Always pass `model` and `effort` explicitly. If an agent type is unknown in this session (the
   file is newer than the session), say so: a new session registers it and is the clean fix. If
   the user continues here, spawn the built-in `Plan` type (it has no Edit, Write or NotebookEdit
   tool; `general-purpose` has them) with the same model and effort, and add to the prompt: "You
   are the reviewer in that file, not a planner: ignore the Plan role and its output format. Use
   only Read, Grep, Glob and the Bash commands the agent file allows; no Skill, MCP, browser,
   web or terminal tools, even if they are listed." Plan carries those tools, so its read-only
   limits rest on that prompt. If a spawn fails for any other reason, say so in the report;
   never present one reviewer's result as the full review.
7. Wait for both reports. Show them together, each finding tagged with its reviewer, most severe
   first; mark findings both reviewers made; list each scenario with its result. Then stop: the
   verify-then-fix step (`docs/setup-playbook.md`, "After a review") comes next and starts from
   these reports. A finding may be rejected only with a reproducing test or a code citation; every
   rejected finding above low is listed for the user with the reason. Never use `--fix`. A fix
   round runs once, on the fix range only, when a fix commit touches a high-stakes file other than
   purely additive test changes (new test functions or files; any modified or deleted test line
   counts as a touch), whatever the severity of the finding it fixed. Findings from that round
   are fixed and listed for the user; a third round needs the user's request.
