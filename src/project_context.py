"""
Project-level instructions inspired by agent/skill systems.

Files are plain Markdown so project teams can understand and edit them:
- AGENTS.md or agent.md: assistant role and boundaries
- workflow.md: project workflow details
- .agents/skills/*/SKILL.md: reusable project-specific skills
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


MAX_CONTEXT_CHARS = 24000


@dataclass
class ProjectContext:
    agent_text: str = ""
    workflow_text: str = ""
    skills: list[tuple[str, str]] = None

    def __post_init__(self):
        self.skills = self.skills or []

    def to_prompt_section(self) -> str:
        sections = []
        if self.agent_text:
            sections.append("## Project Agent Instructions\n" + self.agent_text.strip())
        if self.workflow_text:
            sections.append("## Project Workflow\n" + self.workflow_text.strip())
        if self.skills:
            skill_lines = []
            for name, text in self.skills:
                skill_lines.append(f"### Skill: {name}\n{text.strip()}")
            sections.append("## Project Skills\n" + "\n\n".join(skill_lines))
        combined = "\n\n".join(sections).strip()
        if len(combined) > MAX_CONTEXT_CHARS:
            combined = combined[:MAX_CONTEXT_CHARS] + "\n\n[Project context truncated]"
        return combined


def _read_first_existing(root: Path, names: list[str]) -> str:
    for name in names:
        path = root / name
        if path.exists() and path.is_file():
            return path.read_text(encoding="utf-8", errors="replace")
    return ""


def load_project_context(study_dir: Path) -> ProjectContext:
    root = study_dir.resolve()
    agent_text = _read_first_existing(root, ["AGENTS.md", "agent.md", ".agents/AGENTS.md", ".agents/agent.md"])
    workflow_text = _read_first_existing(root, ["workflow.md", ".agents/workflow.md"])

    skills = []
    skills_dir = root / ".agents" / "skills"
    if skills_dir.exists():
        for skill_path in sorted(skills_dir.glob("*/SKILL.md")):
            text = skill_path.read_text(encoding="utf-8", errors="replace")
            skills.append((skill_path.parent.name, text))

    return ProjectContext(agent_text=agent_text, workflow_text=workflow_text, skills=skills)


def describe_project_context(study_dir: Path) -> str:
    context = load_project_context(study_dir)
    lines = []
    lines.append(f"Agent instructions: {'found' if context.agent_text else 'not found'}")
    lines.append(f"Workflow document: {'found' if context.workflow_text else 'not found'}")
    lines.append(f"Project skills: {len(context.skills)}")
    for name, _text in context.skills:
        lines.append(f"- {name}")
    return "\n".join(lines)


def create_starter_project_context(study_dir: Path) -> list[Path]:
    root = study_dir.resolve()
    created = []

    agent_path = root / "agent.md"
    if not agent_path.exists():
        agent_path.write_text(
            """# Project Agent Role

You are a CSES survey harmonization assistant for this project.

Follow the documented workflow, keep the human reviewer in control, and record
processing decisions clearly in the project log. Do not move to the next step
without explicit user approval.
""",
            encoding="utf-8",
        )
        created.append(agent_path)

    workflow_path = root / "workflow.md"
    if not workflow_path.exists():
        workflow_path.write_text(
            """# Project Workflow Notes

Use this file for project-specific workflow notes that supplement the standard
CSES 16-step process.

## Local conventions

- Add any project-specific file naming, review, or documentation conventions here.

## Known issues

- Add known collaborator, data, language, or documentation issues here.
""",
            encoding="utf-8",
        )
        created.append(workflow_path)

    skill_dir = root / ".agents" / "skills" / "project-review"
    skill_path = skill_dir / "SKILL.md"
    if not skill_path.exists():
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_path.write_text(
            """---
name: project-review
description: Use when reviewing project-specific documentation and decisions.
---

# Project Review Skill

When reviewing this project, compare collaborator documentation against the
workflow notes and record any ambiguity as a focused collaborator question.
""",
            encoding="utf-8",
        )
        created.append(skill_path)

    return created
