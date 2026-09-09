# 中英双语项目页与 GitHub Pages

源文件位于 [`site/`](../site/index.html)。首页连接六个中英双语完整章节：方法、协议、结果、行为、优化与复现。页面支持指标/成功失败/均值中位数 p90 切换、训练 step 滑块、五版本技能展开、13 组研究图表、数值下载和可搜索的源文档资料库。原始 skill 与工程文档保留原语言；不改写冻结策略正文。

## 本地构建

发布图表已入库，网页构建需要 Python 与 Markdown 渲染库，不要求 Node.js、GPU 或运行中的环境服务：

```bash
python -m pip install -e ".[site]"
python scripts/build_site.py
python scripts/check_publication.py
python -m http.server 8000 --directory _site
```

打开 [localhost:8000](http://localhost:8000)。不要直接用 `file://` 打开 HTML，浏览器可能阻止加载 JSON。需要重新计算图表时，先执行分析和绘图命令，见[分析说明](../artifacts/analysis/README.md)。

`_site/` 为被 Git 忽略的构建目录。构建脚本只打包明确列出的网页文件、SVG/PNG、数值数据与 Markdown 资料，不复制 `runs/`、`.env`、模型权重或商品语料。页面使用相对路径，可部署到仓库子路径，正文和图表无需访问私有 GitHub 仓库即可阅读。`build-manifest.json` 记录每个产物的 SHA-256。原始工程文档的公式由固定版本 MathJax 3.2.2 CDN 渲染；无网络时保留可读 TeX，主要内容、图表与交互数据仍在本地。

## GitHub Pages

工作流为 [`.github/workflows/pages.yml`](../.github/workflows/pages.yml)，在默认发布分支 `master` 的相关文件变化时构建，也可手动运行。仓库设置中将 **Settings → Pages → Build and deployment → Source** 设为 **GitHub Actions**。工作流先检查站点，再通过 Pages artifact 部署；只有 `master` 可进入部署 job。

此配置遵循 [GitHub 自定义 Pages 工作流文档](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)：上传静态产物，部署 job 使用 `pages: write`、`id-token: write` 与 `github-pages` environment。仓库访问权限或套餐不支持 Pages 时，需要仓库管理员先处理设置，不能把本地构建成功当成线上发布成功。

按当前 remote `XBQNRCM/ShopSimRL`，默认预期地址为 `https://xbqnrcm.github.io/ShopSimRL/`；以实际部署输出为准。仓库更名或分支变更时需同步页面内的 GitHub 文档链接与工作流。网页没有添加自定义域名、追踪脚本、外部字体依赖或后端服务。

## 内容更新

实验内容先更新数值摘录及分析，重新生成图表后再构建页面。双语章节源码在 `site/chapters/`，行为和优化表格由冻结 JSON 注入。执行 `python scripts/render_analysis_docs.py` 将相同内容同步到中文 docs。四条件概览在 README 和 HTML 中也有静态副本，`check_publication.py` 检查关键数字、双语字段、站内链接/锚点与构建 manifest。更新结果时应同步所有摘要。

跨平台构建时，`.gitattributes` 保留 `artifacts/analysis/` 的原始字节，避免 W&B 导出和 manifest 因换行转换失配。主分析对历史 JSON/CSV 输入使用统一 CRLF→LF 的哈希；gzip 和构建产物仍按原始字节校验。

本次交付的核验包含数据复算、图表图片检查、离线构建和链接/结构检查。未把自动浏览器截图或交互测试当作已完成检查；如后续需要跨浏览器或移动端实际操作测试，可针对现有页面继续验证。
