

if OBJECT_ID('Florence3DActions_23_copy', 'U') IS NOT NULL
    DROP TABLE Florence3DActions_23_copy;
if OBJECT_ID('Florence3DActions_23_origin', 'U') IS NOT NULL
    DROP TABLE Florence3DActions_23_origin;
if OBJECT_ID('Florence3DActions_23_filtered_len_s', 'U') IS NOT NULL
    DROP TABLE Florence3DActions_23_filtered_len_s;
if OBJECT_ID('Florence3DActions_23_test20', 'U') IS NOT NULL
    DROP TABLE Florence3DActions_23_test20;

--创建副本
select * into  Florence3DActions_23_copy from Florence3DActions_23
--复制表结构
SELECT TOP 0 * INTO Florence3DActions_23_origin FROM Florence3DActions_23_copy;
SELECT TOP 0 * INTO Florence3DActions_23_filtered_len_s FROM Florence3DActions_23_copy;



-- 创建测试集20%
IF OBJECT_ID('dbo.Florence3DActions_23_test20', 'U') IS NOT NULL
    DROP TABLE dbo.Florence3DActions_23_test20;
SELECT h.*
INTO Florence3DActions_23_test20
FROM Florence3DActions_23_copy h
JOIN (
   SELECT
    TRIM(REPLACE(REPLACE(REPLACE(REPLACE(value, '''', ''), CHAR(13), ''), CHAR(10), ''), '"', '')) AS FileName
FROM Florence3DExperimentTest
CROSS APPLY STRING_SPLIT(CAST(testSamples20 AS NVARCHAR(MAX)), ',')
) AS t
ON h.FileName = t.FileName
WHERE t.FileName <> '';



select distinct filename,type from Florence3DActions_test20 order by type
select distinct filename,type from Florence3DActions_23_test20 order by type
