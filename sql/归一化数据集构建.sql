--归一化数据集构建
select top 0 * into Florence3DActions_norm20 from Florence3DActions_copy;
select top 0 * into Florence3DActions_norm23 from Florence3DActions_copy;
select top 0 * into UTKinectAction3D_norm60 from UTKinectAction3D_copy;
select top 0 * into MSRAction3D_norm60 from MSRAction3D_copy;
select top 0 * into HanYueDailyAction3D_norm70 from HanYueDailyAction3D_copy;

select top 0 * into test_Florence3DActions_filtered_len_s from Florence3DActions_copy;


truncate table Florence3DActions_norm20
truncate table Florence3DActions_norm23
truncate table UTKinectAction3D_norm60
truncate table MSRAction3D_norm60
truncate table HanYueDailyAction3D_norm70


select distinct type,fileName from HanYueDailyAction3D_norm70 order by filename
select distinct type,fileName from HanYueDailyAction3D_copy order by  filename


select distinct type,fileName from MSRAction3D_norm60 order by filename
select distinct type,fileName from MSRAction3D_copy order by  filename

select distinct type,fileName from Florence3DActions_copy order by filename
select distinct type,fileName from Florence3DActions_norm20 order by  filename


--构建原始训练集
select top 0 * into Florence3DActions_norm20_origin from Florence3DActions_norm20;
select top 0 * into Florence3DActions_norm23_origin from Florence3DActions_norm23;
select top 0 * into UTKinectAction3D_norm60_origin from UTKinectAction3D_norm60;
select top 0 * into MSRAction3D_norm60_origin from MSRAction3D_norm60;
select top 0 * into HanYueDailyAction3D_norm70_origin from HanYueDailyAction3D_norm70;

--构建滤波训练集
select top 0 * into Florence3DActions_norm20_filtered_len_s from Florence3DActions_norm20;
select top 0 * into Florence3DActions_norm23_filtered_len_s from Florence3DActions_norm23;
select top 0 * into UTKinectAction3D_norm60_filtered_len_s from UTKinectAction3D_norm60;
select top 0 * into MSRAction3D_norm60_filtered_len_s from MSRAction3D_norm60;
select top 0 * into HanYueDailyAction3D_norm70_filtered_len_s from HanYueDailyAction3D_norm70;

--构建测试集
-- 构建HanYue70
if(object_id('hanyueDailyAction3D_norm70_test20','U') IS NOT NULL)
    drop table hanyueDailyAction3D_norm70_test20
SELECT h.*
INTO hanyueDailyAction3D_norm70_test20
FROM HanYueDailyAction3D_norm70 h
JOIN (
   SELECT
    TRIM(REPLACE(REPLACE(REPLACE(REPLACE(value, '''', ''), CHAR(13), ''), CHAR(10), ''), '"', '')) AS FileName
FROM HanYueExperimentTest
CROSS APPLY STRING_SPLIT(CAST(testSamples20 AS NVARCHAR(MAX)), ',')
) AS t
ON h.FileName = t.FileName
WHERE t.FileName <> '';

--构建测试集
-- 构建HanYue70
if(object_id('hanyueDailyAction3D_norm100_test20','U') IS NOT NULL)
    drop table hanyueDailyAction3D_norm100_test20
SELECT h.*
INTO hanyueDailyAction3D_norm100_test20
FROM HanYueDailyAction3D_norm100 h
JOIN (
   SELECT
    TRIM(REPLACE(REPLACE(REPLACE(REPLACE(value, '''', ''), CHAR(13), ''), CHAR(10), ''), '"', '')) AS FileName
FROM HanYueExperimentTest
CROSS APPLY STRING_SPLIT(CAST(testSamples20 AS NVARCHAR(MAX)), ',')
) AS t
ON h.FileName = t.FileName
WHERE t.FileName <> '';

--构建UTK60
if(object_id('UTKinectAction3D_norm60_test20','U') IS NOT NULL)
    drop table UTKinectAction3D_norm60_test20
SELECT h.*
INTO UTKinectAction3D_norm60_test20
FROM UTKinectAction3D_norm60 h
JOIN (
   SELECT
    TRIM(REPLACE(REPLACE(REPLACE(REPLACE(value, '''', ''), CHAR(13), ''), CHAR(10), ''), '"', '')) AS FileName
FROM UT3DExperimentTest
CROSS APPLY STRING_SPLIT(CAST(testSamples20 AS NVARCHAR(MAX)), ',')
) AS t
ON h.FileName = t.FileName
WHERE t.FileName <> '';

--构建MSR60
if(object_id('MSRAction3D_norm60_test20','U') IS NOT NULL)
    drop table MSRAction3D_norm60_test20
SELECT h.*
INTO MSRAction3D_norm60_test20
FROM MSRAction3D_norm60 h
JOIN (
   SELECT
    TRIM(REPLACE(REPLACE(REPLACE(REPLACE(value, '''', ''), CHAR(13), ''), CHAR(10), ''), '"', '')) AS FileName
FROM MSR3dexperimentTest
CROSS APPLY STRING_SPLIT(CAST(testSamples20 AS NVARCHAR(MAX)), ',')
) AS t
ON h.FileName = t.FileName
WHERE t.FileName <> '';

--构建Florence20
if(object_id('Florence3DActions_norm20_test20','U') IS NOT NULL)
    drop table Florence3DActions_norm20_test20
SELECT h.*
INTO Florence3DActions_norm20_test20
FROM Florence3DActions_norm20 h
JOIN (
   SELECT
    TRIM(REPLACE(REPLACE(REPLACE(REPLACE(value, '''', ''), CHAR(13), ''), CHAR(10), ''), '"', '')) AS FileName
FROM Florence3DExperimentTest
CROSS APPLY STRING_SPLIT(CAST(testSamples20 AS NVARCHAR(MAX)), ',')
) AS t
ON h.FileName = t.FileName
WHERE t.FileName <> '';

--构建Florence23
if(object_id('Florence3DActions_norm23_test20','U') IS NOT NULL)
    drop table Florence3DActions_norm23_test20
SELECT h.*
INTO Florence3DActions_norm23_test20
FROM Florence3DActions_norm23 h
JOIN (
   SELECT
    TRIM(REPLACE(REPLACE(REPLACE(REPLACE(value, '''', ''), CHAR(13), ''), CHAR(10), ''), '"', '')) AS FileName
FROM Florence3DExperimentTest
CROSS APPLY STRING_SPLIT(CAST(testSamples20 AS NVARCHAR(MAX)), ',')
) AS t
ON h.FileName = t.FileName
WHERE t.FileName <> '';


--查询测试集
select distinct filename,type from hanyueDailyAction3D_norm70_test20 order by type
select distinct filename,type from MSRAction3D_norm60_test20 order by type
select distinct filename,type from UTKinectAction3D_60_test20 order by type
select distinct filename,type from Florence3DActions_norm20_test20 order by type


select distinct type,fileName from HanYueDailyAction3D_norm70_origin order by type

select  count(frameOrder) from HanYueDailyAction3D_norm70_origin group by  fileName
select  count(frameOrder) from HanYueDailyAction3D_norm70_filtered_len_s group by  fileName
select * from HanYueDailyAction3D_norm70_origin order by fileName, frameOrder
select  count(frameOrder) from HanYueDailyAction3D_norm70_filtered_len_s group by  fileName
select * from HanYueDailyAction3D_norm70_origin
select * from florence3DActions_norm23_origin order by fileName, frameOrder



---删除训练集

truncate table Florence3DActions_norm20_filtered_len_s
truncate table Florence3DActions_norm23_filtered_len_s
truncate table UTKinectAction3D_norm60_filtered_len_s
truncate table MSRAction3D_norm60_filtered_len_s
truncate table HanYueDailyAction3D_norm70_filtered_len_s