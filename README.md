# TikTok Live AR Demo · 雨幕与烟花

一个运行在浏览器中的实时直播 AR 原型：用户微笑时触发雨幕，露齿大笑时触发烟花；烟花粒子可与随头部移动和旋转的碰撞边界产生物理反馈。

## Live Demo

[在线体验](https://part2-vibe-ar.vercel.app)

## Source Code

[GitHub 公开仓库](https://github.com/Fliex-jie/zijie-tiktok-live-ar-demo)

## 核心体验

- 微笑触发双层雨幕：视频中景、Canvas 前景雨丝与落地水花；
- 大笑触发程序化烟花，并保留自然的粒子拖尾与衰减；
- 用户主动移动头部撞击指定烟花时，粒子沿碰撞法线反弹；
- 表情信号经过平滑、滞回和持续时间判断，减少误触发与状态抖动；
- 可主动开启声音辅助，麦克风不可用时自动降级为纯视觉模式；
- 支持横屏/竖屏预览和可发送消息的直播聊天区域；
- 摄像头与麦克风数据仅在浏览器本地处理，不上传或保存。

## 技术栈

- TypeScript + Vite
- MediaPipe Face Landmarker
- Canvas 2D 粒子渲染与碰撞响应
- HTML5 Video 雨幕合成
- Web Audio API（可选声音辅助）

## 本地运行

```bash
npm install
npm run dev
```

打开终端输出的本地地址，点击“开始体验”并允许摄像头权限。声音辅助只会在用户主动点击后请求麦克风权限。

## 构建与预览

```bash
npm run build
npm run preview
```

生产环境需要使用 HTTPS，浏览器才会允许网页调用摄像头和麦克风。

## 调试入口

- `?debug=1`：显示面部关键点、碰撞边界和性能信息；
- `?preview=rain`：直接预览雨幕状态；
- `?preview=firework&debug=1`：预览烟花与测试碰撞边界。

## 性能策略

- 摄像头根据设备 CPU、内存提示选择 1080p、720p 或 480p，并根据真实渲染 FPS 与推理耗时自动降级；
- 人脸推理与视觉渲染分频执行；
- 渲染循环以 60 FPS 为上限；
- Safari 仅在雨幕激活时以 480×270、20 FPS 执行透明合成，避免黑幕并限制主线程开销；
- 粒子使用固定容量数组，避免持续增长；
- 同一组人脸关键点复用于表情判断与头部碰撞，避免重复推理；
- MediaPipe 模型与 WASM 版本固定，降低运行时不确定性。
