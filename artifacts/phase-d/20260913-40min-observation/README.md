# 40分钟真实观察证据索引

会话：`aed4a2ef-2915-46c7-a04b-da1a4addbaf7`  
任务：`e6e65acf-7164-49e7-b013-790893ae4f40`  
源数据库：`data/product/observer.sqlite3`

这组文件由只读SQLite事务直接提取。运行 `extract_evidence.py` 可以从当前产品库重新生成其余数据文件；`reproduce-queries.sql` 列出对应原始查询。

## 建议检查顺序

1. `verification-checks.json`：先核对会话/任务均为ended、2400.031秒、95分、5分扣分、9次提醒和资源预算通过。
2. `event-timeline.csv`：逐项核对事件成立、提醒、扣分、恢复、时间偏移、Denver本地时间及提醒交付状态。
3. `signal-totals.csv`：核对五类信号的eligible、valid、unknown、异常累计和各自覆盖率。
4. `resource-summary.json`：核对CPU平均/P95、内存峰值、增长、样本数和预算判定。
5. `runtime-identity-and-policy.json`：核对真实摄像头、模型哈希、个人profile、本次基线、启动耗时及五条规则阈值。
6. `raw-session-records.json`：上述记录的完整合并快照，包括数据库原始action ID和任务字段。
7. `reproduce-queries.sql`：需要独立对照产品库时使用。
8. `SHA256SUMS.txt`：核对导出后文件是否变化。

## 证据边界

- 当时产品截图开关为关闭，evidence表中该会话为0条，因此没有事件图片可以检查。
- 9条alert均为`delivered`且没有delivery_error。这证明Qt主线程完成了横幅显示、系统提示音请求、窗口闪烁调用和状态回写；它不能证明扬声器音量足够或使用者实际听见声音。
- 界面显示的“有效覆盖100%”按整数取整，并且只使用away和phone_held两个硬信号。原始最低硬信号覆盖为99.831%；attention_away单项覆盖为85.665%。
- 累计异常时间包含没有连续达到触发阈值的片段，不能直接当作事件持续时间。
