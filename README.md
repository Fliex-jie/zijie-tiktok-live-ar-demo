# Part 2 · Vibe AR Prototype

当前阶段是 Spike 01：验证浏览器摄像头、MediaPipe Face Landmarker 和实时 Debug HUD 链路。

## 本阶段目标

- 用户点击按钮后请求摄像头权限；
- 在浏览器本地读取视频，不上传或保存画面；
- 以约 12 FPS 调用 Face Landmarker；
- 输出人脸是否存在、笑容分数、张嘴分数、人脸推理耗时和渲染 FPS；
- 为后续微笑下雨、大笑烟花和头部碰撞提供可观测数据。

## 本地运行

```bash
npm install
npm run dev
```

打开终端输出的 localhost 地址，点击“开始体验”，然后允许摄像头权限。

## 注意

- 线上部署必须使用 HTTPS，浏览器才会允许生产域名调用摄像头。
- 本阶段还没有实现最终的雨滴、烟花、状态机和物理碰撞。
- MediaPipe 模型和 WASM 文件当前使用固定版本的远程资源；后续部署时需要再次验证资源加载和缓存策略。
