# References: how to work with Claude Code on this project

Collected 2026-10-10 from Anthropic's published guidance. The playbook (`docs/setup-playbook.md`)
already follows most of it; this file is the short version with links, for the start of a session.

## The practices, and where this repo applies them

| Practice (Anthropic) | How this repo does it |
| --- | --- |
| Explore, plan, implement, commit. Do not code first; have Claude read, propose a plan you review, then implement. Plan mode exists for this; small clear fixes do not need it. | The design document gets its own session, started in plan mode. Every later item starts from `docs/plan.md` and a `progress.json` item. |
| Give Claude a way to verify its own work (tests, lints, types it runs itself), and a second pass by a different model that tries to refute the result. | `scripts/check.sh` at the start and end of every session; tiered review with two reviewer models (`docs/setup-playbook.md`, "Code review playbook"). |
| Keep state in files, not chat: commit often with descriptive messages and keep a progress file. | `docs/plan.md`, `docs/progress.json`, `docs/progress-log.md`; one commit per item. |
| CLAUDE.md short and specific: build commands, standards, decisions. Too long and half of it is ignored. It is context, not enforcement: anything that must never happen belongs in a hook or permission rule. | CLAUDE.md under about 80 lines pointing at the docs; the data guard is a git hook, CI stage and Read deny rule, not a sentence. |
| Start from a clean git state and checkpoint so a bad run can be reverted. | Session start checks `git status` and CI; one item per session. |
| Specific prompts: name the file, the scenario, the test that should pass. | Each `progress.json` item carries a description and verifiable steps. |
| Manage context: performance degrades as the window fills; fresh sessions per task. | One item per session; `/clear` between tasks; the design document in a dedicated session. |

## Reading list

- Best practices for Claude Code (docs): https://code.claude.com/docs/en/best-practices
- Claude Code best practices (engineering blog): https://www.anthropic.com/engineering/claude-code-best-practices
- How Anthropic teams use Claude Code: https://www.anthropic.com/news/how-anthropic-teams-use-claude-code
- How Claude remembers your project (CLAUDE.md, memory): https://code.claude.com/docs/en/memory
- Common workflows (plan mode, worktrees, subagents): https://code.claude.com/docs/en/common-workflows
- Effective context engineering for AI agents: https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
- Effective harnesses for long-running agents: https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents
- How AI is transforming work at Anthropic: https://www.anthropic.com/research/how-ai-is-transforming-work-at-anthropic

## Course material

- Lecture repo (Zero to Mastery PyTorch, fork): https://github.com/skillspire-github/pytorch-deep-learning
  Chapters this project follows: 04 custom datasets, 06 transfer learning, 09 deployment.
- Class guideline: `docs/course-guideline.md`.
