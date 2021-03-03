SELECT
CAST(MIN(INSDATE) AS Date) AS start_date,
CAST(MAX(INSDATE) AS Date) AS end_date,
COUNT(*) AS records,
WSG_ID,
WZD_ID,
ST_ID,
AT_ID
FROM dbo.prozessmessung_schleifen
WHERE WARM = {warm}
AND TEACH_ACTIVE = {teach_active}
AND READY = {ready}
AND MA_NR = {ma_nr}
GROUP BY WSG_ID, WZD_ID, ST_ID, AT_ID
HAVING COUNT(*) > 1000
ORDER BY records DESC