# Reusable playbook: a tiered, user-started code review workflow for Claude Code

Written 2026-10-10 from the Animal Spirits project. Nothing here depends on machine learning:
the same design fits a website, an API, a mobile app or a data pipeline. Replace the examples
with your project's own rules. Part A explains the design, Part B is a prompt you paste into a
new project to have Claude set it up.

## Part A: the design

### A1. Principle

Spend review effort in proportion to the cost of a missed bug. Most commits need a cheap review
or none; a few touch files where a bug costs money, data, users or trust. A script decides which
is which from the changed paths, so the decision does not depend on anyone's memory or mood.

### A2. Five layers, each catching what the one before misses

1. **Tests and a single check script.** Any rule a test can check gets a test. The script runs
   tests, type checks, lint, format and any guards (secrets, data files) and is the only thing
   that runs in CI. Check its exit code before every commit.
2. **Tier selection.** `scripts/risk_files.sh <base>..<head>` prints the changed files that match
   a high-stakes regex. Empty output means a cheap review is enough.
3. **Reviewers, started by the user.** Claude reports the tier, the matching invariants and the
   exact command; the user types it or says skip. Claude never starts a review on its own.
4. **Verify, then fix.** Each finding is checked against the code before anything changes;
   real ones are fixed with a regression test that fails on the old code.
5. **Measure and update.** When a bug slips through, its invariant row and scenario are updated
   and the log records which tier missed it.

### A3. Tiers

| Tier | When | Command |
|---|---|---|
| 0 | No code changed (docs, content, config of no consequence) | nothing |
| 1 | Code changed, `risk_files.sh` printed nothing | `/code-review xhigh --max-findings all <base>..HEAD` |
| 2 | High-stakes files changed | `/review-session` (two reviewers on different models) |
| 3 | A milestone that cannot be undone | tier 2 plus `/code-review ultra`, scheduled with time to fix |
| run | A manual action with no diff (a deploy, a migration run, a paid job, a bulk email, a data export) | review the record the code wrote, not a diff (see A6) |

One fix round, on the fix range only, runs when a fix commit touches a high-stakes file other
than purely additive test changes, whatever the severity of the finding it fixed; a further round
needs the user's request (otherwise every regression test triggers the next round). Schedule
tier 3 where a fix round still fits before the deadline, not at the very end.

Two lessons from the first review of this setup: list the config files that silently override
your main tool config (`pytest.ini`, `tox.ini`, `setup.cfg`, `mypy.ini`, `ruff.toml`, any
`conftest.py`; for JavaScript, `jest.config.*`, `.eslintrc*`, `tsconfig.*`) as high-stakes, or a
new one can turn off a check with no tracked file touched; and add a test that parses the
invariant map's "where" column and asserts the regex matches every entry, so the map and the
script cannot drift apart.

### A4. What counts as high-stakes (adapt the list)

Code where a bug costs money, leaks data, breaks the law or a platform's terms, or cannot be
undone, plus the control plane that protects it. For a website that is typically:

- authentication, sessions, password reset, OAuth callbacks
- payments, billing, webhooks from the payment provider, pricing and tax logic
- anything that reads or writes personal data; exports; email and notification senders
- database schema and migrations; data deletion and retention jobs
- authorization checks, admin routes, feature flags that gate access
- deployment and infrastructure config, secrets handling, CORS and CSP headers
- the control plane: tests, test config, the check script, CI workflows, git hooks, lint
  config that could silence a rule, and the Claude agent and permission files themselves

Put the list in one regex in `scripts/risk_files.sh` with a test that asserts a sample of risky
and safe paths. Add a new high-stakes module in the same commit that creates it.

### A5. The invariant map (`docs/review/invariants.md`)

One row per rule that must always hold: id, rule, source document and section, where it can
break (paths and work items), a scenario for a reviewer to trace, and the test that checks it or
"none yet". Empty test cells are the visible gaps. Write it by hand from the rules you already
have (requirements, compliance notes, platform terms, architecture decisions); a 30-40 row table
is enough. Website examples:

| id | rule | scenario to trace |
|---|---|---|
| R01 | No secrets, tokens or personal data in logs, error pages, analytics events or commits | An exception handler logs the request body; a 500 page shows the stack trace in production |
| R02 | Every route that changes data checks authentication and the caller's permission on the object, not only the role | An admin endpoint checks `is_admin` but not ownership; a webhook route skips signature checks |
| R03 | Payment amounts are computed server-side from the catalogue, never trusted from the client; webhooks are idempotent | A retry of the same webhook creates two orders; a discount is applied twice |
| R04 | Migrations are reversible and backward compatible with the running version | A column is dropped while the old deploy still reads it |
| R05 | Deletion requests remove the data everywhere it was copied: caches, search index, backups schedule, exports | A user is deleted from the DB; their email stays in the mailing-list sync |
| R06 | Accessibility and privacy commitments stated to users hold (cookie consent before tracking, alt text, keyboard navigation) | Analytics fires before consent on the landing page |
| R30 | A skip, xfail, narrowed test path or lint ignore never removes a check a row above relies on | A failing auth test gets `@skip` to fix CI |

Scenarios are passed to the reviewers as focus, not as limits: one reviewer always reviews the
whole range freely.

### A6. Actions with no diff

Deploys, migration runs, paid API jobs, bulk emails, exports and manual admin actions are where
the costly mistakes happen, and a diff-based review never sees them. Make the code write a record
(a ledger table or append-only file: what ran, with what parameters, estimated and actual cost,
who approved) and make the record the unit of review: the log entry for the action quotes the
ledger, and a test asserts the guard (for example, a job over the remaining budget is refused
before it is submitted, not reconciled after).

### A7. Reviewer design (what the evidence supported)

- **Two reviewers on different models**, one generalist with the whole range and one specialist
  on the high-stakes files with the session's scenarios. In this repo every reviewer missed
  something another one caught, and two reviewers had no overlapping findings.
- **Fresh context.** Reviewers see the diff and the rules, never the implementer's reasoning or
  the narrative log (anchoring). Known gaps to skip are passed as a short list of ids.
- **Read-only by instruction and by permission.** Deny `Read` on data and env files in
  `.claude/settings.json`; reviewer agents get only Read, Grep, Glob and Bash for git, tests and
  lint. They never edit, commit or run anything against production.
- **Severity defined in the prompt.** "High" means an invariant breaks, money is spent wrongly or
  data leaks; everything else is low. Reviewers asked to find gaps always find some; a defined bar
  and "nothing found is a valid result" keep the noise down.
- **Short base prompt plus matched scenarios.** A long checklist dilutes the rules that matter.
  Focus works; checklists alone add little (human-reviewer studies; treat as an assumption for
  LLM reviewers).
- **Union of findings, each verified.** No majority voting across runs. A finding may be rejected
  only with a reproducing test or a code citation, and every rejected finding above low is listed
  for the user with the reason, because the verifier is usually the implementer.
- **Never `--fix`.** Report first. An auto-fix edits whatever worktree runs it, outside the
  session's checkpoints, and skips the verification step.
- **Pass model and effort explicitly on every spawn.** Agent files load at session start; a
  worktree created from an old commit loads stale ones. In the session that creates a new agent
  file, spawn a general-purpose agent and tell it to follow the file.
- **Built-in `/code-review` stays blind to the map.** It follows CLAUDE.md only, so it is one
  reviewer that does not share the map's blind spots. Type the effort level every time (a bare
  call reuses the last typed level, even from another session) and pass a range.

### A8. Session protocol around it

Start: check the branch base (`git merge-base --is-ancestor main HEAD`), run the check script,
look at the last CI run. Work: one item per commit. End: check script green, everything
committed, report the tier and wait. After the review: verify, fix with tests, log "Review: tier
N, ran X" or "skipped by user". Merge and push only with the user's go-ahead. State lives in
files (a progress log and a plan), not in chat.

### A9. Known limits

Nothing blocks a skipped review; the log line is the record and the user is the control point.
The path regex is the gatekeeper for tier selection. Reviewer agents with Bash are read-only by
instruction. The map is as complete as the rules it was built from. None of this has been
measured as a catch rate; it is a design that makes coverage visible and misses traceable.

## Part B: prompt to set this up in a new project

Paste the text below into Claude Code in the new project (adjust the bracketed parts first).

```text
Set up a tiered, user-started code review workflow for this project, following the design in
the attached playbook (Part A). Do it in one branch, one commit per step, and show me each file
before committing. Do not run any review yourself; I start reviews.

Context: this project is [a Next.js e-commerce site / a Django API / ...]. The costly mistakes
here are [payments, auth, personal data, deployments, ...]. Rules the project must keep live in
[README, docs/requirements.md, privacy policy, platform terms, ...]. Deadline: [date]. I pay per
token, so prefer short prompts and few reviewer runs.

Steps:
1. Read the project's rule documents and write docs/review/invariants.md: 30-40 rows with id,
   rule, source and section, where it can break (paths, work items), a scenario to trace, and the
   test that checks it or "none yet". Include a control-plane row (skips, lint ignores, narrowed
   test paths) and a row for every action that happens without a diff (deploys, migrations,
   paid jobs, bulk sends, exports) saying what record the code must write.
2. Write scripts/risk_files.sh (`<base>..<head>` in, matching changed paths out, exit 2 on a bad
   range) with a regex covering every path in the map's "where" column plus tests/, test config,
   the check script, CI, hooks, lint config and .claude/. Add tests/test_risk_files that assert a
   sample of risky and safe paths, renames shown under the old path, and non-ASCII paths unquoted.
3. Write .claude/agents/code-reviewer.md (generalist, whole range, model [opus], effort [xhigh])
   and .claude/agents/risk-reviewer.md (specialist, given files, a different model: [fable or
   another], effort [high]). Both: tools Read, Grep, Glob, Bash; read-only; never read env or data
   files; never read the progress log narrative; trace the scenarios the caller passes and report
   holds / breaks / could not check; severity defined (high = an invariant breaks, money is spent
   wrongly or data leaks); output = findings most severe first with file:line, failure scenario,
   how verified, suggested fix; then what was checked and found sound.
4. Write .claude/commands/review-session.md: refuse with uncommitted changes; range =
   merge-base(main, HEAD)..HEAD unless given; run risk_files.sh; select map rows by path prefix or
   passed item ids; flag "spec changed" rows when a cited document is in the range; report the
   tier (0 nothing / 1 `/code-review xhigh --max-findings all <base>..HEAD` / 2 both agents in
   parallel with model and effort passed explicitly / run items: ledger check); merge the two
   reports, tag each finding with its reviewer, mark overlaps, list each scenario's result; then
   stop. Fix round whenever a fix touches a high-stakes file, on the fix range only.
5. Add deny rules in .claude/settings.json for Read on env files and any data directories, and
   allow rules for the test, lint and git read commands.
6. Add to CLAUDE.md a "Review" step: at session end report the tier, the matching invariants, the
   exact command and a rough cost, then wait; never start a review unsolicited; a finding may be
   rejected only with a test or a code citation, listed for me; fix with regression tests; never
   use --fix; log "Review: tier N, ran X" or "skipped by user"; merge and push only with my
   go-ahead; pass model and effort explicitly on every spawn.
7. Run the check script, then show me `bash scripts/risk_files.sh main..HEAD` on this branch so
   I can start its own first review.
```
