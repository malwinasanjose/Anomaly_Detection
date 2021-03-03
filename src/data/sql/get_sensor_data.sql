SELECT
ps.[PRIMARY],
{sensors}
FROM dbo.prozessmessung_schleifen ps
WHERE WARM = {warm}
AND TEACH_ACTIVE = {teach_active}
AND READY = {ready}
AND MA_NR = {ma_nr}
AND WSG_ID = {wsg_id}
AND WZD_ID = {wzd_id}
AND ST_ID = {st_id}
AND AT_ID = {at_id}
AND INSDATE BETWEEN '{start_date}' AND '{end_date}'
