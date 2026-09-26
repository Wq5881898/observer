-- Run against data/product/observer.sqlite3
.headers on
.mode box
SELECT * FROM sessions WHERE id='aed4a2ef-2915-46c7-a04b-da1a4addbaf7';
SELECT * FROM tasks WHERE id='e6e65acf-7164-49e7-b013-790893ae4f40';
SELECT * FROM totals WHERE session_id='aed4a2ef-2915-46c7-a04b-da1a4addbaf7' ORDER BY signal;
SELECT offset,rule_id,kind,points,reason,delivery_state,delivery_attempted,delivery_error
FROM actions WHERE session_id='aed4a2ef-2915-46c7-a04b-da1a4addbaf7' ORDER BY offset,rowid;
SELECT * FROM evidence WHERE session_id='aed4a2ef-2915-46c7-a04b-da1a4addbaf7';
