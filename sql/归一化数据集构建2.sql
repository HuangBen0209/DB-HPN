
SELECT TOP 0 * INTO Florence3DActions_norm30 FROM Florence3DActions_copy;
SELECT TOP 0 * INTO UTKinectAction3D_norm40 FROM UTKinectAction3D_copy;
SELECT TOP 0 * INTO MSRAction3D_norm40 FROM MSRAction3D_copy;
SELECT TOP 0 * INTO HanYueDailyAction3D_norm100 FROM HanYueDailyAction3D_copy;


SELECT TOP 0 * INTO Florence3DActions_norm30_origin FROM Florence3DActions_copy;
SELECT TOP 0 * INTO UTKinectAction3D_norm40_origin FROM UTKinectAction3D_copy;
SELECT TOP 0 * INTO MSRAction3D_norm40_origin FROM MSRAction3D_copy;
SELECT TOP 0 * INTO HanYueDailyAction3D_norm100_origin FROM HanYueDailyAction3D_copy;


drop table if exists Florence3DActions_norm30;
drop table if exists UTKinectAction3D_norm40;
drop table if exists MSRAction3D_norm40;
drop table if exists HanYueDailyAction3D_norm100;

drop table if exists Florence3DActions_norm30_origin;
drop table if exists UTKinectAction3D_norm40_origin;
drop table if exists MSRAction3D_norm40_origin;
drop table if exists HanYueDailyAction3D_norm100_origin;


--构建测试集
-- 构建HanYue_norm100
if(object_id('hanyueDailyAction3D_norm100_test20','U') IS NOT NULL)
    drop table hanyueDailyAction3D_norm100_test20
SELECT h.*
INTO hanyueDailyAction3D_norm100_test20
FROM hanyueDailyAction3D_norm100 h
JOIN (
   SELECT
    TRIM(REPLACE(REPLACE(REPLACE(REPLACE(value, '''', ''), CHAR(13), ''), CHAR(10), ''), '"', '')) AS FileName
FROM HanYueExperimentTest
CROSS APPLY STRING_SPLIT(CAST(testSamples20 AS NVARCHAR(MAX)), ',')
) AS t
ON h.FileName = t.FileName
WHERE t.FileName <> '';



--构建UTK_norm40
if(object_id('UTKinectAction3D_norm40_test20','U') IS NOT NULL)
    drop table UTKinectAction3D_norm40_test20
SELECT h.*
INTO UTKinectAction3D_norm40_test20
FROM UTKinectAction3D_norm40 h
JOIN (
   SELECT
    TRIM(REPLACE(REPLACE(REPLACE(REPLACE(value, '''', ''), CHAR(13), ''), CHAR(10), ''), '"', '')) AS FileName
FROM UT3DExperimentTest
CROSS APPLY STRING_SPLIT(CAST(testSamples20 AS NVARCHAR(MAX)), ',')
) AS t
ON h.FileName = t.FileName
WHERE t.FileName <> '';

--构建MSR_norm40
if(object_id('MSRAction3D_norm40_test20','U') IS NOT NULL)
    drop table MSRAction3D_norm40_test20
SELECT h.*
INTO MSRAction3D_norm40_test20
FROM MSRAction3D_norm40 h
JOIN (
   SELECT
    TRIM(REPLACE(REPLACE(REPLACE(REPLACE(value, '''', ''), CHAR(13), ''), CHAR(10), ''), '"', '')) AS FileName
FROM MSR3dexperimentTest
CROSS APPLY STRING_SPLIT(CAST(testSamples20 AS NVARCHAR(MAX)), ',')
) AS t
ON h.FileName = t.FileName
WHERE t.FileName <> '';

--构建Florence_norm30
if(object_id('Florence3DActions_norm30_test20','U') IS NOT NULL)
    drop table Florence3DActions_norm30_test20
SELECT h.*
INTO Florence3DActions_norm30_test20
FROM Florence3DActions_norm30 h
JOIN (
   SELECT
    TRIM(REPLACE(REPLACE(REPLACE(REPLACE(value, '''', ''), CHAR(13), ''), CHAR(10), ''), '"', '')) AS FileName
FROM Florence3DExperimentTest
CROSS APPLY STRING_SPLIT(CAST(testSamples20 AS NVARCHAR(MAX)), ',')
) AS t
ON h.FileName = t.FileName
WHERE t.FileName <> '';