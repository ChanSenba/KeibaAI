import re

from keibaai import factor_view, gui_help


def test_tooltip_links_point_to_help_sections():
    anchors = {a for a, _, _ in gui_help.HELP} | {f"factors-{i}" for i in range(len(factor_view.SECTIONS))}
    linked = {m for t in gui_help.TIPS.values() for m in re.findall(r"help#([\w-]+)", t)}
    assert linked and linked <= anchors


def test_help_anchors_are_unique():
    anchors = [a for a, _, _ in gui_help.HELP]
    assert len(anchors) == len(set(anchors))
