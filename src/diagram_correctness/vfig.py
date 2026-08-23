from __future__ import annotations

import subprocess
from pathlib import Path


class VFigRunner:
    """Run the official VFIG single-image inference command.

    The command is an argument array, not a shell string. Supported placeholders
    are ``{image}`` and ``{svg}``.
    """

    def __init__(self, command: list[str], timeout_seconds: int = 600) -> None:
        if not command:
            raise ValueError("VFIG command cannot be empty")
        self.command = command
        self.timeout_seconds = timeout_seconds

    def convert(self, image: str | Path, output_svg: str | Path) -> Path:
        image_path = Path(image).resolve()
        output_path = Path(output_svg).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            argument.format(image=str(image_path), svg=str(output_path))
            for argument in self.command
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "VFIG failed with exit code "
                f"{completed.returncode}: {completed.stderr.strip() or completed.stdout.strip()}"
            )
        if not output_path.exists():
            raise RuntimeError(f"VFIG did not create the expected SVG: {output_path}")
        return output_path

