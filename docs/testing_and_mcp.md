> 基于代码核对(2026-08-20),仅记录代码/配置/测试已确认的事实,运行行为以源码为准。

# 测试体系与 MCP 监控服务

本项目主链路是 JSON 进、JSON 出的四阶段流水线。测试体系优先保护阶段产物契约,其次保护纯函数规则,并用 fake 后端隔离模型、GPU 与网络。MCP 服务提供只读观测 + 最小管理,供 AI 客户端查询进程与失败运行。

## 一、测试体系

### 1. tests/ 三层布局与职责

- `tests/contract/`(`tests/contract/*.py`,5 个文件):阶段产物契约测试,覆盖 demo/phase2/phase3a/phase3b/phase4。职责是加载 `fixtures/` 黄金样本,调用 `tests.contracts` 校验器断言返回 `[]`,并测负面/边界用例(如 `tests/contract/test_phase3b_contract.py:10` 拒绝已删除情绪、`tests/contract/test_phase3a_contract.py:10` 允许静默回合)。不调模型、不依赖 GPU。
- `tests/unit/`：纯函数、规则函数与轻量模块单测。当前 Git 中可按 `git.exe ls-files tests/unit` 复核文件数；内容覆盖切窗、hype、对齐、demo 查询、prompt 注入、service_manager 健康判定以及云端调用层。
- 配音任务单相关测试（历史实施记录，计划文件已不在当前 Git 跟踪内容中）：
  - `test_speech_measure.py`/`test_speech_profile.py`:speech_units_v1 特征(中文/英文名/数字/C4/标点/情绪标签)、时长估算与安全上界、profile 指纹/状态、离线标定拆分/拟合/验收。
  - `test_rule_fact_ids.py`/`test_rule_neutral_renderer.py`:原子事实 ID 稳定性(数组重排/derived origin/去重/碰撞 fail-closed/legacy topic 隔离)、纯模板 neutral/capsule(排序/连接词/禁截断/required 覆盖)。
  - `test_voice_task_contract.py`(计划书 §19 文件清单):neutral v4 / commentary v3 / final voice task 三个纯结构校验器的 fixture 正例与负例(契约黄金样本 `tests/fixtures/phase{3,4}/neutral_v4.json|commentary_v3.json|final_voice_task.json`)。
  - `test_voice_task_selection.py`:Phase4 v2/v3 双契约、惰性选稿(primary 一次 TTS / 逐级降级 / render_unfit 阻止发布)、profile 指纹不匹配、固定 tick 越界。
  - `test_phase3b_v3_tasks.py`:Phase3b 稀疏候选(green 无 compact / amber 至多两次调用 / red 预判直出 compact / 终判降级 / profile 未 validated 回退 v2 / preserved 由校验器计算)。
  - `test_preflight_v3v4.py`:preflight 按 schema_version 分流校验 neutral v4/commentary v3/final manifest、统一语音计量口径(`count_spoken_chars`,不再用 `len()` 重算)。
  - 契约测试扩展:`tests/contract/test_phase3a_contract.py`(v4 正负例)、`test_phase3b_contract.py`(commentary v2/v3)、`test_phase4_contract.py`(final voice task)。
- 调用层/速度优化新增测试(2026-08-16):
  - `test_llm_shim.py`:协议层/分流入口,含阶段 0 诊断字段(usage_tokens/in_flight/queue_ms/retry_category/validation_reason)、阶段 1 护栏(超时拆分、scope 信号量、SSE 总时限、不做 thinking 调配)。拆分后 monkeypatch 目标为 `llm_protocol`/`cloud_adapter`/`local_adapter`。
  - `test_env_secrets.py`:`_load_secrets` 新密钥合同(profile→scope→fallback、旧键回退、warnings)。
  - `test_cloud_cache.py`:成功响应缓存 18 用例(pending→confirm、TTL 清理、键敏感性、并发原子写、命中同构)。
  - `test_phase3a_concurrency.py` / `test_phase3b_concurrency.py`:LLM-A 两阶段并发 / LLM-B 三段式并发(缺省串行兼容、并发产物等价、熔断)。
  - `test_config_profile.py`:单份配置无 profile 覆盖层、冲突标量拒绝。
  - `test_call_layer_adapters.py`:adapter 边界（cloud 拒绝回环/local 拒绝远端、SSE 只进 cloud、非流式只进 local、两端均不注入 thinking 调配字段、薄入口按 loopback 分流）。
- Phase4 严格执行与媒体同步：`test_phase3bc_render_v2.py`、`test_phase4_execution_v2.py`、`test_phase4_media_gate.py`、`test_media_clock.py`、`tests/contract/test_phase4_sync_contract.py`；覆盖 v2 交接、final_text 所有权、PCM sample 边界、媒体发布状态和 fail-closed。
- `tests/` 顶层：历史遗留测试,原地保留(`readme_test.md:17`)。文件数以 `git.exe ls-files 'tests/test_*.py'` 复核；多为工具链/脚本级集成测试,常用 `importlib.util` 动态加载 `tools/` 脚本、`subprocess` 跑 CLI `--help` 断言输出,或断言某些文件已被删除(`tests/test_vllm_runtime_tools.py:56`)。
- `tests/fixtures/`:小型黄金样本 JSON/JSONL,分 `demo/`、`phase2/`、`phase3/`、`phase4/` 子目录。约定单样本 <20KB、禁放密钥与绝对路径(`readme_test.md:23`)。
- `tests/conftest.py`:将项目根加入 `sys.path`(`tests/conftest.py:12`);用显式文件映射注入业务 marker(`:59`);提供 fixture `fixtures_dir`(`:77`)、`load_fixture`(`:82`,`.jsonl` 按行解析,统一 `utf-8-sig` 解码)、`fake_backends`(`:93`)。
- `tests/contracts.py`:阶段契约校验器 + CLI(见第 3 节)。
- `tests/fakes.py`:零网络 fake 后端(见第 2 节)。
- 包结构:`tests/__init__.py` 存在(0 字节),故支持 `python -m tests.contracts` 与 `from tests.contracts import ...`;`mcp/` 目录无 `__init__.py`,按脚本方式启动(`mcp/server.py:8`)。

### 2. marker 与联网/GPU/真实模型禁令

- 业务 marker 定义于 `pytest.ini`,由 `tests/conftest.py:59` 按文件名显式映射:`runtime`、`demo_phase1`、`phase2`、`phase3a`、`phase3b`、`phase4`、`mcp`、`data_tools`;跨模块/CLI/子进程/事务 I/O 测试另标 `integration`。`ffmpeg` 与 `slow` 仍用于环境/耗时隔离。
- 禁令落地靠 `tests/fakes.py`,而非 marker:
  - `FakeLLM`(`tests/fakes.py:10`):预置响应列表轮流返回,记录 prompts/calls。
  - `FakeVLM`(`tests/fakes.py:31`):固定返回视觉描述文本。
  - `FakeTTS`(`tests/fakes.py:44`):写一段静音 WAV 到磁盘,不调真实 TTS。
- `fake_backends` fixture(`tests/conftest.py:93`)用 `monkeypatch.setattr` 把 `sbmachine.llma_api.generate`、`sbmachine.llmb_api.generate`、`audio_service.gpt_sovits_client.synthesize` 替换为 fake,从而隔离网络/GPU/真实模型。
- 约束:测试默认不联网、不调真实模型、不占 GPU(`readme_test.md:70`、`README.md:526`)。需要真实模型/ffmpeg/耗时环境的测试必须显式打 marker(`readme_test.md:48`)。

### 3. tests/contracts.py CLI 真实用法

入口 `main(argv)`(`tests/contracts.py:119`),`python -m tests.contracts` 与直接运行均触发。

- 位置参数 `stage`:choices = `demo`、`phase2`、`phase3a`、`phase3b`、`phase4`、`all`(`tests/contracts.py:121`,即 `STAGE_VALIDATORS` 键 + `all`)。
- 可选参数:`--file <路径>`(单文件,支持 `.json`/`.jsonl`)、`--output-dir <目录>`。
- 分支逻辑:
  - `all` 必须带 `--output-dir`,否则报错(`tests/contracts.py:127`)。按 `DEFAULT_STAGE_FILES`(`:124-128`)检查 phase2=`rounds_with_yolo.json`、phase3a=`rounds_with_neutral.json`、phase3b=`commentary.json`、phase4=`rounds_final.json`;若 `<output-dir>/demo/` 存在则加验 demo;缺文件报 `missing <filename>`。
  - 单阶段:优先用 `--file`;例外是 `demo` 可用 `--output-dir` 当 demo 目录(`:133`);其余单阶段无 `--file` 会报错(`:136`)。
- 退出码:任一阶段有错误返回 1 并打印 `[FAIL] <stage>` + 逐条错误,全通过打印 `[OK]` 返回 0(`tests/contracts.py:138`)。
  - 校验器分工:`validate_neutral`(phase3a)、`validate_commentary`(phase3b)在本文件内定义(`tests/contracts.py:23`、`:61`); v4 neutral 有独立 `validate_neutral_v4`，但当前 `STAGE_VALIDATORS["phase3a"]` 仍固定指向 v3，CLI 不会按 schema 自动分流。`demo`/`phase2`/`phase4` 复用 `sbmachine.preflight` 的对应校验器。phase3b 允许情绪值仅 `{平述, 激动, 惊叹}`(`tests/contracts.py:14`)。

### 4. 常用测试命令

- 快速 CPU 全量:`python -m pytest tests/ -v -m "not ffmpeg and not slow"`(`README.md:497`、`readme_test.md:55`)。
- 契约 + 单元:`python -m pytest tests/contract tests/unit -v`(`README.md:503`)。
- 只跑契约:`python -m pytest tests/contract -v`(`readme_test.md:75`)。
- Phase2 快速单元:`python -m pytest tests -q -m "phase2 and not integration"`。
- Phase3a 快速单元:`python -m pytest tests -q -m "phase3a and not integration"`。
- Phase3b 快速单元:`python -m pytest tests -q -m "phase3b and not integration"`。
- Phase4 严格链路：`python -m pytest tests/unit/test_phase3bc_render_v2.py tests/unit/test_phase4_execution_v2.py tests/unit/test_phase4_media_gate.py tests/unit/test_media_clock.py tests/contract/test_phase4_sync_contract.py -q`。
- 黑盒生产数据门禁:`python -m pytest tests/unit/test_production_gates.py -v`（从历史成功产物检查 neutral+style 文本；运行稳定性与失败原因需以当前复测为准）。
- 调用层/速度优化定点:`python -m pytest tests/unit/test_llm_shim.py tests/unit/test_env_secrets.py tests/unit/test_cloud_cache.py tests/unit/test_config_profile.py tests/unit/test_phase3a_concurrency.py tests/unit/test_phase3b_concurrency.py -q`。
- MCP 单元:`python -m pytest tests -q -m "mcp"`。
- 单阶段契约体检:`python -m tests.contracts phase2 --file output/sbmachine/rounds_with_yolo.json`(`README.md:509`)。
- 全目录契约体检:`python -m tests.contracts all --output-dir output/sbmachine/`(`README.md:516`)。
- 语法检查:`python -m compileall -q sbmachine core tools tests`(`README.md:523`)。

## 二、MCP 监控服务

服务定义于 `mcp/server.py`,`server = FastMCP("ai6657-monitor")`(`mcp/server.py:45`),定位为只读观测 + 最小管理:唯一写操作是 `kill_process`。

MCP 安全边界由 `tests/unit/test_mcp_server.py` 的 19 个离线单元测试保护,覆盖进程匹配/拒杀、HTTP 与 Compose 降级、legacy manifest、失败目录排序、路径逃逸与诊断日志 tail。测试注入最小 FastMCP 注册桩,不要求根环境安装 MCP 依赖,也不启动服务。

### 5. 六个工具签名与行为

进程判定口径(`mcp/server.py:32`、`_match_project` `:65`):
- `STRONG_KEYWORDS`(sbmachine/parse_demo/gpt-sovits/api_v2.py/llamafactory/llama-factory)单独成立,返回 `strong_keyword`。
- `GENERIC_KEYWORDS`(run.py/vllm/ffmpeg/python)必须叠加本仓库路径证据:cmdline 含 REPO 路径(`generic+repo_path`)或 cwd 在 REPO 下(`generic+repo_cwd`);否则不匹配。

- `list_processes(filter: str = "") -> dict`(`mcp/server.py:109`):遍历 `psutil.process_iter()`,排除自身及父进程链(`_self_and_ancestor_pids` `:55`),按 `_match_project` 判定,`filter` 对 cmdline 大小写不敏感子串过滤。返回 `{processes, count, note}`,每项含 pid/name/cmdline/cwd/matched_by/status/cpu_percent(0.1s 采样)/memory_mb/create_time。容器内进程对 psutil 不可见(`mcp/server.py:9`)。
- `check_services() -> dict`(`mcp/server.py:191`):见第 6 节。HTTP 探测 `_probe_http`(`:138`)的存活口径:`urlopen(timeout=3)` 成功,或收到 `HTTPError`(含 401/404)都判为 `http_reachable=True`(服务端有响应即视为进程存活),仅 `URLError`/`OSError`/`ValueError` 为 False。另跑 `docker compose ps --all --format json`(`_compose_states` `:149`,兼容整体数组与 NDJSON);docker 不可用时 `compose_state=null`、`docker_available=false`,不报错。
- `get_last_run() -> dict`(`mcp/server.py:211`):只读转发 `output/sbmachine/run_manifest.json`。不存在返回 `{manifest_exists: False}`;解析失败返回 `manifest_exists:True + parse_error`;成功抽取 run_id/status/publishable/enabled_phases/checkpointed_stages/raw_path。
- `list_error_runs(limit: int = 10) -> dict`(`mcp/server.py:261`):扫描 `output/error/`(source=error)与 `output/.staging/`(source=staging)下的 run 子目录(`_scan_run_dirs` `:234`),按 mtime 倒序,截取 `max(1, limit)`。每项含 run_id/source/mtime + 读 `failure.json` 的 failed_stage/error(`_failure_fields` `:246`);无 failure.json 时对应字段为 null。
- `get_error_detail(run_id, file="", tail_lines=200) -> dict`(`mcp/server.py:321`):`_resolve_run_dir`(`:285`)在 error/staging 两根定位并防逃逸(run_id 不得含 `/`、`\`、`..`,resolve 后须在根下)。`file` 留空返回 failure.json + 文件清单(`_list_run_files` `:298`:顶层文件 + `diagnostics/` 递归,排除 `checkpoints/`、`publish/` 子树);带 `file` 时三重校验(不得绝对路径/`..`/`:`、resolve 后须在 run_dir 下、路径段不得命中排除子树),读取后仅返回尾部 `tail_lines`(默认 200)行,超 `MAX_CONTENT_CHARS=100_000`(`:42`)再截断并置 `truncated`。
- `kill_process(pid: int, force: bool = False) -> dict`(`mcp/server.py:380`,唯一写操作):三道安全门控 —— ① pid 属于自身/父进程链直接拒绝(`:387`);② 进程不存在/无法访问返回失败(`:389`);③ 必须通过 `_match_project` 判定,否则拒绝(`:396`)。通过后 `force=True` 走 `proc.kill()`,否则 `proc.terminate()`,返回 `{pid, killed, reason, matched_by}`。

### 6. check_services 覆盖范围

`SERVICES`(`mcp/server.py:36`)仅探测两个:`talk_service`(vLLM,`:8000`,`/v1/models`)与 `audio_service`(GPT-SoVITS,`:9880`,`/openapi.json`),口径与 `sbmachine/service_manager.py:_health_url`(`:72`)的 vllm/sovits 一致。`service_manager` 另有 vlm 健康 URL(`:23333/health`,`sbmachine/service_manager.py:80`),但 `check_services` 未纳入探测。

### 7. 注册与依赖隔离

- `.mcp.json`(根目录):注册 server `ai6657-monitor`,`type: stdio`,`command: python`,`args: ["mcp/server.py"]` —— 脚本方式启动、stdio 传输(对应 `mcp/server.py:410` 的 `server.run()`)。
- 依赖隔离:`mcp/requirements.txt` 仅 `mcp` 与 `psutil`;根 `requirements.txt` 不含二者,互不影响。安装:`pip install -r mcp/requirements.txt`。
- `mcp/` 目录禁放 `__init__.py`(`mcp/server.py:8`),以脚本形式运行。

## 已知偏差

`ffmpeg`/`slow` marker 仍在 `pytest.ini` 与 `tests/conftest.py` 重复声明,且当前零测试使用;业务 marker 已实际用于分单元运行。原 `fake_backends` 未构造 `FakeVLM` 的问题已修复。

