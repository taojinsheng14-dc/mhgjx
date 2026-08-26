# 手机自动化工具箱

这是一个基于 Python、ADB 和 scrcpy 的可视化手机自动化工具箱。模板由使用者自行添加，流程按照“截图识别 -> 自然点击/滑动 -> 随机等待 -> 下一步”编排。

## 启动

手机开启 USB 调试并连接电脑后，双击：

```text
启动工具箱.bat
```

也可以在 PowerShell 中启动：

```powershell
cd C:\Users\tjs\Documents\脚本\scrcpy_human_automation
.\.venv\Scripts\python.exe .\toolbox.py
```

## 使用方法

1. 点击“刷新截图”，确认左侧出现当前手机画面。
2. 在手机画面上拖动鼠标框选按钮或图标，点击“将选区保存为模板”。
3. 也可以进入“模板库”，直接导入已有的 PNG/JPG 图片。
4. 在“流程编排”中添加“识别点击”“随机等待”或“自然滑动”。
5. 调整步骤顺序和循环次数，点击“运行流程”。
6. 运行中可随时点击“停止”。

模板保存在 `templates/`，默认流程自动保存在 `flows/default.json`。模板库允许为空，不需要项目预置 `start_button.png`。

## 识别建议

- 只截取目标按钮或图标的稳定部分，不要包含时间、头像、动画等会变化的内容。
- 默认阈值 `0.88` 适合画面变化较小的场景；识别困难时可逐步调低到 `0.80` 左右。
- 先使用“测试识别”查看当前画面的最高匹配度，再决定阈值。

## 项目结构

```text
toolbox.py                         可视化工具箱
启动工具箱.bat                    双击启动入口
templates/                        用户模板库
flows/default.json                自动保存的流程
src/scrcpy_human_automation/      自动化核心代码
examples/template_click_flow.py   Python API 示例
```

`scrcpy` 用于手机投屏和人工观察，自动化输入与截图通过 ADB 执行。即使没有把 `scrcpy` 加入 PATH，只要 ADB 设备已连接，工具箱的截图、识别和操作仍可正常工作。

## 视频流测试说明

当前 `厢房_视频流测试` 是独立测试流程，不会影响正式厢房/抓鬼/飞贼流程。它使用 Android `screenrecord` H264 流 + PyAV 解码来减少每次 ADB 截图耗时。

注意：这只是过渡验证方案，最终目标是接入 `scrcpy` 内部视频流。scrcpy 内部视频流理论上更适合长期实时识别，但实现复杂度更高，后续应继续以独立测试流程接入，确认稳定后再考虑替换正式流程。
