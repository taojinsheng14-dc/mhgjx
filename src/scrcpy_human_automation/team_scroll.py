import json
import re
import time
import cv2
from datetime import date

from .frame_source import AdbFrameSource
from .humanizer import RelativeRegion


def today_count(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        value = data["completed"]
        if data["date"] == date.today().isoformat() and type(value) is int and value >= 0:
            return value
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return None


def click_scrolled_card(r, step):
    src = AdbFrameSource(r.device, cache_ttl=0)
    region = RelativeRegion.from_ltrb
    def find(image, name, box, threshold):
        return r.matcher.find_template(r.templates_dir / name, screenshot=image,
                                       search_region=region(*box), threshold=threshold)

    def team_page(image):
        if not find(image, "活动弹窗标题_每日活动.png", (.37, .02, .64, .13), .84):
            return False
        # Selected team tab is dark blue; inactive tabs are pale blue.
        h, w = image.shape[:2]
        patch = image[int(.166*h):int(.19*h), int(.685*w):int(.715*w)]
        b, g, red = patch.mean(axis=(0, 1))
        return b - red > 65 and red < 145

    if r.stop_event.is_set():
        return
    r._interruptible_delay(.18, .32)
    first = src.latest().image
    if not team_page(first):
        r._handle_timeout(step, "未确认组队页已选中，不滑动")
        return
    def position(im):
        text = r._ocr_image_detail(im, region(.099, .069, .174, .126), .65)
        pairs = re.findall(r"(?<!\d)(\d{1,3})\s*[,，]\s*(\d{1,3})(?!\d)", text)
        if len(pairs) == 1:
            return ("xy", *map(int, pairs[0]))
        visible = re.fullmatch(r"[（(]\s*(\d{1,3})\s*[)）]?", text.strip())
        return ("visible", int(visible[1])) if visible else None

    baseline = position(first)
    if r.stop_event.is_set():
        return
    # Start inside the list, above its bottom toolbar.
    from .humanizer import RelativePoint
    start = r.human.to_absolute(RelativePoint(.56, .68))
    end = r.human.to_absolute(RelativePoint(.56, .36))
    r.log(f"组队列表向上滑一次：{start} -> {end}")
    r.device.swipe(*start, *end, r.human.random.randint(320, 440))
    r._interruptible_delay(.50, .68)
    if r.stop_event.is_set():
        return
    def locate(image):
        if not team_page(image):
            return None
        title = find(image, step["template"], (.33, .22, .815, .745), step.get("threshold", .84))
        if not title:
            return None
        target = title.region
        left_col = (target.left + target.right)/2 < .57
        y = (target.top + target.bottom)/2
        box = (.49 if left_col else .735, max(.225, y-.05),
               .568 if left_col else .812, min(.745, y+.05))
        if box[3]-box[1] >= .045:
            for n in ((1, 3, 5) if left_col else (2, 4, 6)):
                button = find(image, f"参加{n}.png", box, .88)
                if button:
                    return title, button

    previous = None
    for _ in range(5):
        if r.stop_event.is_set():
            return
        image = src.latest().image
        pair = locate(image)
        if pair and previous and all(abs(a-b) <= 3 for m,n in zip(pair, previous)
                                    for a,b in zip(m.pixel_box,n.pixel_box)):
            title, button = pair
            point = r._tap_region(button.region, pre_delay=tuple(step["pre_delay"]),
                                  post_delay=tuple(step["post_delay"]), tap_mode="swipe",
                                  padding_ratio=step.get("padding_ratio", .03))
            if r.stop_event.is_set():
                return
            sent = time.monotonic()
            r.log(f"滑后双帧稳定，已发点击（未确认出发）：任务={title.pixel_box} 参加={button.pixel_box} 点={point}")
            evidence = image.copy()
            for match in pair:
                x,y,x2,y2 = match.pixel_box
                cv2.rectangle(evidence,(x,y),(x2,y2),(0,255,255),2)
            cv2.circle(evidence,point,8,(0,0,255),2)
            r.failures_dir.mkdir(exist_ok=True)
            ok, png = cv2.imencode(".png", evidence)
            if ok:
                png.tofile(str(r.failures_dir / f"team_click_{time.time_ns()}.png"))
            if baseline is None:
                if r.ghost_safety:
                    r.ghost_safety.participate_sent_at = sent
                r.log("点击前人物坐标不可读，不推断出发或失败；进入带传送保护的接任务等待")
                return
            moved = 0
            while time.monotonic()-sent < 14 and not r.stop_event.is_set():
                r._interruptible_delay(.35,.55)
                if r.stop_event.is_set():
                    return
                fresh = src.latest().image
                if r.ghost_safety and r.ghost_safety.role_match(fresh,"accept"):
                    r.ghost_safety.participate_sent_at = sent
                    r.log("已出现钟馗接任务按钮，确认到达")
                    return
                pos = position(fresh)
                changed = baseline and pos and pos[1] != baseline[1]
                if baseline and pos and baseline[0] == pos[0] == "xy":
                    changed = pos != baseline
                moved = moved+1 if changed else 0
                if moved >= 2:
                    if r.ghost_safety:
                        r.ghost_safety.participate_sent_at = sent
                    r.log(f"连续两帧人物坐标已变化：{baseline} -> {pos}")
                    return
            if not r.stop_event.is_set():
                raise TimeoutError(f"参加已点，未确认人物出发（原坐标={baseline}）；保留现场，不关窗、不补点")
            return
        previous = pair
        r._interruptible_delay(.15,.25)
    if not r.stop_event.is_set():
        r._handle_timeout(step, "滑动后未确认钟馗任务及同行参加按钮，不点击")
