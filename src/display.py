"""Simple box and rule formatting for terminal output."""

import textwrap

WIDTH = 72

# Box-drawing characters
TOP_LEFT = "\u250c"
TOP_RIGHT = "\u2510"
BOTTOM_LEFT = "\u2514"
BOTTOM_RIGHT = "\u2518"
HORIZ = "\u2500"
VERT = "\u2502"
T_LEFT = "\u251c"
T_RIGHT = "\u2524"


def rule(char: str = "\u2500") -> str:
    """Return a horizontal rule spanning WIDTH."""
    return char * WIDTH


def box(title: str, content: str, wrap_width: int = WIDTH - 4) -> str:
    """Wrap content and draw a box with an optional title. Content is word-wrapped."""
    lines = []
    for para in content.split("\n"):
        lines.extend(textwrap.wrap(para, width=wrap_width) or [""])
    border = HORIZ * (wrap_width + 2)
    out = [TOP_LEFT + border + TOP_RIGHT]
    if title:
        out.append(VERT + " " + title[: wrap_width].ljust(wrap_width) + " " + VERT)
        out.append(T_LEFT + HORIZ * (wrap_width + 2) + T_RIGHT)
    for line in lines:
        out.append(VERT + " " + line.ljust(wrap_width) + " " + VERT)
    out.append(BOTTOM_LEFT + border + BOTTOM_RIGHT)
    return "\n".join(out)
