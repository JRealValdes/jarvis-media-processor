"""``jmp fixtures`` commands."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from jarvis_media_processor.config import get_settings
from jarvis_media_processor.fixtures import (
    FFmpegNotFoundError,
    generate_scenario,
    generate_scenarios,
    list_scenario_ids,
)

logger = logging.getLogger(__name__)

fixtures_app = typer.Typer(help="Generate synthetic base/donor fixtures for tests.")


@fixtures_app.command("list")
def fixtures_list() -> None:
    """List available synthetic scenario ids."""
    for scenario_id in list_scenario_ids():
        typer.echo(scenario_id)


@fixtures_app.command("generate")
def fixtures_generate(
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Directory for generated scenarios."),
    ] = Path("fixtures/synthetic"),
    scenario: Annotated[
        list[str] | None,
        typer.Option(
            "--scenario",
            "-s",
            help="Scenario id to generate (repeatable). Default: all.",
        ),
    ] = None,
) -> None:
    """Generate synthetic media pairs and ground_truth.json files."""
    settings = get_settings()
    ids = scenario or list_scenario_ids()
    try:
        if len(ids) == 1:
            truth = generate_scenario(ids[0], output_dir / ids[0], settings=settings)
            results = {ids[0]: truth}
        else:
            results = generate_scenarios(output_dir, ids, settings=settings)
    except FFmpegNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    except (KeyError, RuntimeError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    for scenario_id in results:
        typer.echo(f"generated {scenario_id} -> {output_dir / scenario_id}")
