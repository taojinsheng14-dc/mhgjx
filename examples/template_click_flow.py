from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scrcpy_human_automation import (  # noqa: E402
    AdbDevice,
    AutomationContext,
    AutomationFlow,
    HumanizedController,
    RelativePoint,
    TemplateMatcher,
)


TEMPLATE_PATH = ROOT / "templates" / "start_button.png"


def build_flow(device: AdbDevice) -> AutomationFlow:
    human = HumanizedController(device)
    matcher = TemplateMatcher(device)

    flow = AutomationFlow("template-click-demo")

    def capture(ctx: AutomationContext) -> None:
        ctx.set("screenshot", matcher.screenshot())

    def find_start_button(ctx: AutomationContext) -> None:
        screenshot = ctx.require("screenshot")
        match = matcher.find_template(TEMPLATE_PATH, threshold=0.88, screenshot=screenshot)
        if not match:
            raise RuntimeError(f"Template not found: {TEMPLATE_PATH}")
        ctx.set("start_button", match)

    def click_start_button(ctx: AutomationContext) -> None:
        match = ctx.require("start_button")
        x, y = human.random_tap(match.region.expand(0.05))
        ctx.set("clicked_point", (x, y))

    def swipe_up(ctx: AutomationContext) -> None:
        path = human.natural_swipe(
            start=RelativePoint(0.52, 0.78),
            end=RelativePoint(0.48, 0.28),
            duration_ms=(800, 1300),
        )
        ctx.set("swipe_path", path)

    flow.add_step("capture", capture)
    flow.add_step("find_start_button", find_start_button)
    flow.add_step("click_start_button", click_start_button)
    flow.add_step("swipe_up", swipe_up)
    return flow


if __name__ == "__main__":
    device = AdbDevice()
    device.ensure_connected()
    result = build_flow(device).run()
    print("Flow finished.")
    print(f"Last step: {result.get('last_step')}")
    print(f"Clicked point: {result.get('clicked_point')}")
