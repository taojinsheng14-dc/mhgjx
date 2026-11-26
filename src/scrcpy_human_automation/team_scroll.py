import json
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
    if not team_page(src.latest().image):
        r._handle_timeout(step, "未确认组队页已选中，不滑动")
        return
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
    image = src.latest().image
    if not team_page(image):
        r._handle_timeout(step, "滑动后未确认组队页，不点击")
        return
    title = find(image, step["template"], (.33, .22, .815, .745), step.get("threshold", .84))
    if title:
        target = title.region
        left_col = (target.left + target.right)/2 < .57
        y = (target.top + target.bottom)/2
        box = (.49 if left_col else .735, max(.225, y-.05),
               .568 if left_col else .812, min(.745, y+.05))
        if box[3]-box[1] >= .045:
            for n in ((1, 3, 5) if left_col else (2, 4, 6)):
                button = find(image, f"参加{n}.png", box, .88)
                if button and not r.stop_event.is_set():
                    point = r._tap_region(button.region, pre_delay=tuple(step["pre_delay"]),
                                          post_delay=tuple(step["post_delay"]),
                                          padding_ratio=step.get("padding_ratio", .03))
                    r.log(f"滑后定位：任务框={title.pixel_box} 同行参加框={button.pixel_box} 点击={point}")
                    return
    if not r.stop_event.is_set():
        r._handle_timeout(step, "滑动后未确认钟馗任务及同行参加按钮，不点击")
