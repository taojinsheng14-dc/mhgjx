from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from .frame_source import AdbFrameSource
from .humanizer import RelativeRegion

if TYPE_CHECKING:
    from .runner import WorkflowRunner


class GhostRecoveryRequired(Exception):
    pass


class GhostSafety:
    """Opt-in guards for the single-captain video ghost flow.

    Only the workflow thread sends input. Recovery uses fresh device snapshots,
    not a negative result from the video cache, to select its next action.
    """

    def __init__(self, runner: WorkflowRunner, steps: list[dict], options: dict):
        self.r = runner
        self.steps = steps
        self.options = options
        self.roles = {s["ghost_role"]: i for i, s in enumerate(steps) if s.get("ghost_role")}
        required = {"rest", "activity", "daily", "accept", "close_activity", "eye", "route", "fight", "battle", "flag", "station"}
        if required - self.roles.keys():
            raise ValueError(f"抓鬼恢复配置缺少步骤角色：{sorted(required - self.roles.keys())}")
        self.reset_round()

    def reset_round(self) -> None:
        self.recoveries = 0
        self.close_at = 0.0
        self.participate_sent_at = None
        self.close_checks = 0
        self.close_retry_at = 0.0
        self.fly_on_sent = False
        self.flag_open_requested = False
        self.battle_finished = False

    def step(self, role: str) -> dict:
        return self.steps[self.roles[role]]

    def fresh(self):
        if self.r.stop_event.is_set():
            raise InterruptedError("Workflow stopped")
        return AdbFrameSource(self.r.device, cache_ttl=0).latest().image

    def match(self, image, template: str, region, threshold=0.84):
        return self.r.matcher.find_template(
            self.r.templates_dir / template, screenshot=image,
            threshold=threshold,
            search_region=RelativeRegion.from_ltrb(*region),
        )

    def role_match(self, image, role: str):
        step = self.step(role)
        return self.match(image, step["template"], step["search_region"], step.get("threshold", 0.84))

    def battle(self, image):
        return self.role_match(image, "battle")

    def activity(self, image):
        return self.role_match(image, "activity")

    def normal_hud(self, image):
        return self.match(image, "活动按钮.png", [0.24, 0.0, 0.36, 0.15], 0.80)

    def flag_window(self, image):
        return self.match(image, "flag_map_title.png", [0.15, 0.0, 0.90, 0.40], 0.84)

    def task_text(self, image):
        step = self.step("eye")
        region = RelativeRegion.from_ltrb(*step["search_region"])
        return self.r._ocr_image_detail(image, region, float(step.get("min_confidence", 0.4)))

    def fail(self, step: dict, message: str) -> None:
        self.r._handle_timeout(step, message)

    def before(self, step: dict) -> bool:
        role = step.get("ghost_role")
        if role in {"rest", "activity"}:
            image = self.r.frame_source.latest(max_age=0.3).image
            if self.battle(image) and self.battle(self.fresh()):
                self.fail(step, "开始地面操作前检测到战斗，转入战斗等待")
                return False
            if self.activity(image) and self.activity(self.fresh()):
                self.r.log("活动框已经打开，跳过打坐/重复打开活动")
                return False
        if role == "close_activity":
            delay = self.r.human.random.uniform(*step["pre_delay"])
            self.close_at = (self.participate_sent_at or time.monotonic()) + delay
            self.participate_sent_at = None
            self.close_checks = 0
            self.close_retry_at = self.close_at
            self.r.log(f"活动关闭已并入接任务等待，传送保护{delay:.2f}s；只在确认弹窗存在时点击")
            return False
        if role == "legacy_close":
            return False
        if role == "eye":
            self.open_eye(step)
            return False
        if role == "station":
            self.require_flag_window(step)
        return True

    def after(self, step: dict) -> None:
        role = step.get("ghost_role")
        if role == "accept":
            self.battle_finished = False
            self.r._progress_recorded.clear()
        if role == "fly_on" and self.r.flags.get("need_fly_mount_for_ghost"):
            self.fly_on_sent = True
        if role == "fly_off" and self.r.flags.get("need_fly_mount_for_ghost"):
            self.fly_on_sent = False
        if role == "flag" and self.r.flags.get("fly_station_after_battle"):
            self.flag_open_requested = True
        if role == "station" and self.r.flags.get("fly_station_after_battle"):
            self.verify_station(step)

    def wait_accept(self, step: dict) -> None:
        # The old close delay and accept wait now overlap; no extra fixed sleep.
        deadline = max(time.monotonic(), self.close_at) + float(step.get("timeout", 120))
        while not self.r.stop_event.is_set() and time.monotonic() < deadline:
            frame = self.r.frame_source.latest(max_age=0.4)
            image = frame.image
            now = time.monotonic()
            popup = self.activity(image)
            if now >= self.close_at and popup and now >= self.close_retry_at:
                if self.close_checks >= 3:
                    self.fail(step, "活动弹窗关闭3次仍存在，停止继续点击")
                    return
                # Each retry must be justified by a new screenshot after the last click.
                image = self.fresh()
                if self.battle(image):
                    self.fail(step, "接任务期间检测到战斗，需要恢复战斗等待")
                    return
                if self.activity(image):
                    click = dict(self.step("close_activity"))
                    click["pre_delay"] = [0.12, 0.25]
                    click["debug_screenshot_after"] = False
                    self.r._region_click(click)
                    self.close_checks += 1
                    self.close_retry_at = time.monotonic() + self.r.human.random.uniform(0.6, 0.9)
                    self.r.log(f"活动弹窗已确认存在，关闭尝试 {self.close_checks}/3")
                    continue
            if now >= max(self.close_at, self.close_retry_at) and not self.activity(image):
                match = self.role_match(image, "accept")
                if match:
                    x, y = self.r._click_match(match, step)
                    self.close_at = 0.0
                    self.r.log(f"接任务识别成功 {match.confidence:.3f}，点击 ({x},{y})")
                    return
            self.r._interruptible_wait(float(step.get("interval", 0.25)))
        if not self.r.stop_event.is_set():
            self.fail(step, "接任务等待超时")

    def nonbattle_item(self, image):
        return self.match(image, "天眼通符.png", [0.40, 0.68, 0.90, 0.91], 0.84)

    def confirm_battle_end(self) -> bool:
        started = time.perf_counter()
        image = self.fresh()
        captured = time.perf_counter()
        # All workers only read this same fresh frame; none may send device input.
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="battle-end") as pool:
            battle = pool.submit(self.battle, image)
            hud = pool.submit(self.normal_hud, image)
            item = pool.submit(self.nonbattle_item, image)
            in_battle, has_hud, has_item = battle.result(), hud.result(), item.result()
        self.r.log(
            f"战斗结束复核: 回合={bool(in_battle)} 活动={bool(has_hud)} "
            f"天眼={bool(has_item)} 截图={(captured-started)*1000:.0f}ms "
            f"并行识别={(time.perf_counter()-captured)*1000:.0f}ms"
        )
        if in_battle:
            self.r.log("结束候选被ADB复核否决：回合仍存在，继续战斗等待")
            return False
        if not (has_hud or has_item):
            self.r.log("回合暂未出现，活动按钮和道具栏天眼均未确认；继续等待，不点击")
            return False
        self.r.log("战斗结束已ADB复核：回合消失，活动按钮或道具栏天眼出现")
        self.battle_finished = True
        return True

    def eye_icon(self, image, step: dict):
        region = RelativeRegion.from_ltrb(*step["click_region"])
        # Find the icon near the configured box, but never move clicks outside it.
        match = self.match(image, "天眼通符.png", [0.45, 0.68, 0.64, 0.91], 0.80)
        if match is None:
            return None
        overlap = RelativeRegion.from_ltrb(
            max(region.left, match.region.left), max(region.top, match.region.top),
            min(region.right, match.region.right), min(region.bottom, match.region.bottom),
        )
        return overlap if overlap.width > 0.003 and overlap.height > 0.003 else None

    def open_eye(self, step: dict) -> None:
        image = self.r.frame_source.latest(max_age=0.3).image
        before = self.task_text(image)
        if self.r._looks_like_opened_eye_coordinate(before):
            # Confirm already-opened tasks independently before skipping a consumable.
            before = self.task_text(self.fresh())
            if self.r._looks_like_opened_eye_coordinate(before):
                self.r._set_flags_from_ocr_detail(step, before)
                self.r.log("当前任务已有精确坐标，跳过重复开眼")
                return
        coord = self.r._extract_ghost_coordinate(before)
        box = self.eye_icon(image, step)
        if coord is None or box is None:
            image = self.fresh()
            before = self.task_text(image)
            coord = self.r._extract_ghost_coordinate(before)
            box = self.eye_icon(image, step)
        if coord is None or not self.r._looks_like_ghost_coordinate(before):
            self.fail(step, f"开眼前未确认抓鬼任务及坐标，OCR={before!r}")
            return
        if box is None:
            self.fail(step, "天眼图标未在配置点击范围内确认；可能道具栏移位或道具用尽，不盲点")
            return
        self.r.log(f"开眼前坐标={coord}，天眼图标与用户点击范围交集已确认")
        for attempt in range(2):
            if self.r.stop_event.is_set():
                return
            click = dict(step, type="region_click", region=[box.left, box.top, box.right, box.bottom])
            self.r._region_click(click)
            deadline = time.monotonic() + float(step.get("verify_timeout", 3.2))
            while not self.r.stop_event.is_set() and time.monotonic() < deadline:
                detail = self.task_text(self.r.frame_source.latest(max_age=0.3).image)
                after = self.r._extract_ghost_coordinate(detail)
                if after is not None and after != coord and self.r._looks_like_opened_eye_coordinate(detail):
                    # A previous task's precise coordinate may remain in the video buffer.
                    confirmed = self.task_text(self.fresh())
                    confirmed_coord = self.r._extract_ghost_coordinate(confirmed)
                    if confirmed_coord != coord and self.r._looks_like_opened_eye_coordinate(confirmed):
                        self.r._set_flags_from_ocr_detail(step, confirmed)
                        self.r.log(f"开眼确认成功 {coord}->{confirmed_coord}，ADB复核通过")
                        return
                self.r._interruptible_wait(float(step.get("verify_interval", 0.25)))
            if self.r.stop_event.is_set():
                return
            image = self.fresh()
            detail = self.task_text(image)
            after = self.r._extract_ghost_coordinate(detail)
            if self.r._looks_like_opened_eye_coordinate(detail) and after != coord:
                self.r._set_flags_from_ocr_detail(step, detail)
                self.r.log(f"开眼ADB兜底确认成功 {coord}->{after}")
                return
            box = self.eye_icon(image, step)
            # Only retry when the same unopened task and the same icon are visible.
            if attempt == 0 and after == coord and "附近" in detail and box is not None and self.normal_hud(image):
                self.r.log("ADB确认原坐标未变且天眼仍在范围内，允许重试1次")
                continue
            self.fail(step, f"开眼未确认成功，停止重复消耗；before={coord}, after={after}, OCR={detail!r}")
            return

    def require_flag_window(self, step: dict) -> None:
        if not self.r.flags.get("fly_station_after_battle"):
            return
        deadline = time.monotonic() + 2.0
        while not self.r.stop_event.is_set() and time.monotonic() < deadline:
            if self.flag_window(self.r.frame_source.latest(max_age=0.3).image):
                # A positive cached title is not permission to click an old window.
                if self.flag_window(self.fresh()):
                    self.r.log("合成旗窗口已确认，允许在用户驿站范围内点击")
                    return
            self.r._interruptible_wait(0.15)
        if not self.r.stop_event.is_set():
            self.fail(step, "未确认合成旗窗口，禁止点击驿站坐标")

    def empty_task_list(self, image, detail: str) -> bool:
        # Require visible normal HUD and two untouched inventory controls too.
        # An unrecognised overlay must not become permission for a reset click.
        if not self.normal_hud(image) or self.activity(image) or self.flag_window(image):
            return False
        if not ("师门任务" in detail and "新的一天" in detail and "鬼" not in detail):
            return False
        return bool(
            self.match(image, "打坐.png", [0.72, 0.69, 0.86, 0.87], 0.80)
            and self.match(image, "飞行.png", [0.65, 0.70, 0.76, 0.87], 0.80)
        )

    def normal_world(self, image) -> bool:
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="world-check") as pool:
            hud = pool.submit(self.normal_hud, image)
            item = pool.submit(self.nonbattle_item, image)
            has_hud, has_item = bool(hud.result()), bool(item.result())
        self.r.log(f"正常画面复核：活动={has_hud} 道具栏天眼={has_item}")
        return has_hud or has_item

    def verify_station(self, step: dict) -> None:
        image = self.fresh()
        if self.flag_window(image):
            self.fail(step, "点击驿站后合成旗窗口仍存在，需要有限恢复")
        elif self.battle(image) or self.activity(image) or not self.normal_world(image):
            self.fail(step, "点击驿站后尚未确认正常游戏画面")
        else:
            self.flag_open_requested = False
            self.r.log("驿站点击后确认：合成旗窗口消失，正常游戏画面恢复")

    def recover(self, failed_index: int, message: str) -> int:
        limit = int(self.options.get("max_recoveries", 2))
        if self.recoveries >= limit:
            raise TimeoutError(f"本轮恢复已达{limit}次：{message}")
        self.recoveries += 1
        self.r.log(f"开始状态恢复 {self.recoveries}/{limit}：{message}")
        self.r.device.ensure_connected()
        image = self.fresh()
        if self.battle(image):
            if self.battle_finished:
                raise TimeoutError("结束后再次检测到战斗，无法确定是否同一场，暂停人工处理")
            self.r.log("恢复到战斗等待，不执行地面点击")
            return self.roles["battle"]
        if self.role_match(image, "fight"):
            self.r.log("鬼的战斗对话已出现，恢复到战斗按钮")
            return self.roles["fight"]
        if self.flag_window(image):
            if not self.flag_open_requested:
                raise TimeoutError("存在合成旗窗口，但无法确认由当前流程打开，暂停人工处理")
            self.r.flags["fly_station_after_battle"] = True
            self.r.log("当前流程的合成旗窗口仍在，恢复驿站点击")
            return self.roles["station"]
        if self.activity(image):
            # Safe close is performed by the transit-aware wait, never a floor tap.
            if failed_index >= self.roles["accept"] and self.close_at:
                self.close_checks = 0
                return self.roles["accept"]
            if failed_index < self.roles["accept"]:
                self.r.log("活动框已打开，恢复日常活动页签")
                return self.roles["daily"]
            click = dict(self.step("close_activity"), pre_delay=[0.15, 0.3], post_delay=[0.35, 0.6])
            self.r._region_click(click)
            image = self.fresh()
            if self.activity(image) or self.battle(image):
                raise TimeoutError("恢复关闭活动框后状态仍不确定，暂停人工处理")
        if self.role_match(image, "accept"):
            self.r.log("钟馗接任务对话已出现，恢复接取")
            return self.roles["accept"]
        if not self.normal_world(image):
            raise TimeoutError("未匹配到活动按钮或道具栏天眼，无法确认正常画面，暂停人工处理（不代表已识别出弹框）")
        detail = self.task_text(image)
        if self.r._looks_like_ghost_coordinate(detail):
            self.r._set_flags_from_ocr_detail(self.step("eye"), detail)
            if self.r._looks_like_opened_eye_coordinate(detail):
                # The flying toggle cannot be visually verified: never repeat it here.
                if not self.fly_on_sent:
                    self.r.flags["need_fly_mount_for_ghost"] = False
                self.r.log("已确认抓鬼精确坐标，恢复任务寻路；不补点飞行开关")
                return self.roles["route"]
            if failed_index == self.roles["eye"]:
                raise TimeoutError("已接任务但开眼校验未通过；可能为坐标文字识别、图标或效果确认失败，不代表道具异常，不重复整段开眼")
            self.r.log("已确认未开眼的抓鬼任务，恢复开眼")
            return self.roles["eye"]
        # A missing OCR word is not proof of an empty task list.
        second_image = self.fresh()
        second = self.task_text(second_image)
        if not (self.empty_task_list(image, detail) and self.empty_task_list(second_image, second)):
            raise TimeoutError("未能可靠确定是否已有抓鬼任务，暂停人工处理")
        if failed_index >= self.roles["accept"]:
            if self.fly_on_sent:
                raise TimeoutError("无抓鬼任务但飞行状态无法确认，暂停人工处理，不补点飞行开关")
            self.r.log("两次确认无抓鬼任务，尝试飞棋返回后重新接取")
            self.r.flags["fly_station_after_battle"] = True
            self.r._region_click(self.step("flag"))
            self.flag_open_requested = True
            self.require_flag_window(self.step("station"))
            self.r._region_click(self.step("station"))
            self.verify_station(self.step("station"))
        self.close_at = 0.0
        self.r.log("恢复从活动开始；已有任务不会执行该路径")
        return self.roles["activity"]
