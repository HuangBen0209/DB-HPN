-- 重新构建复制表
if(OBJECT_ID('hanyueDailyAction3D_70_copy') IS NOT NULL)
    drop table hanyueDailyAction3D_70_copy;
-- 从原始表复制所有数据进来
select * into hanyueDailyAction3D_70_copy from hanyueDailyAction3D_70

---------重构表格列名字段
--添加 actiontypename 列
ALTER TABLE hanyueDailyAction3D_70_copy
ADD actiontypename NVARCHAR(100);

UPDATE hanyueDailyAction3D_70_copy
SET actiontypename = sub.type
FROM hanyueDailyAction3D_70_copy AS main
JOIN (
    SELECT filename, type
    FROM hanyueDailyAction3D_70_copy
    GROUP BY filename, type
) AS sub ON main.filename = sub.filename;

--改变原有的type列的内容
UPDATE c
SET c.type = CAST(dt.tablekey AS NVARCHAR(100))
FROM hanyueDailyAction3D_70_copy c
JOIN dict_HanYue_type dt ON c.actiontypename = dt.type;
--复制表结构
if (OBJECT_ID('hanyueDailyAction3D_70_origin','U') IS NOT NULL)
    drop table hanyueDailyAction3D_70_origin
if (OBJECT_ID('hanyueDailyAction3D_70_filtered_len_s','U') IS NOT NULL)
    drop table hanyueDailyAction3D_70_filtered_len_s
SELECT TOP 0 * INTO hanyueDailyAction3D_70_origin FROM hanyueDailyAction3D_70_copy
SELECT TOP 0 * INTO hanyueDailyAction3D_70_filtered_len_s FROM hanyueDailyAction3D_70_copy

-- 创建测试集20%

if(object_id('hanyueDailyAction3D_70_test20','U') IS NOT NULL)
    drop table hanyueDailyAction3D_70_test20
SELECT h.*
INTO hanyueDailyAction3D_70_test20
FROM hanyueDailyAction3D_70_copy h
JOIN (
   SELECT
    TRIM(REPLACE(REPLACE(REPLACE(REPLACE(value, '''', ''), CHAR(13), ''), CHAR(10), ''), '"', '')) AS FileName
FROM HanYueExperimentTest
CROSS APPLY STRING_SPLIT(CAST(testSamples20 AS NVARCHAR(MAX)), ',')

) AS t
ON h.FileName = t.FileName
WHERE t.FileName <> '';


select distinct filename,type from hanyueDailyAction3D_70_test20 order by type
select distinct filename,type from hanyueDailyAction3D_test20 order by type


select top(0) * into HanYueDailyAction3D_norm100 from HanYueDailyAction3D_copy


select * from HanYueExperimentTest















SELECT TOP 0 * INTO HanYueDailyAction3D_norm100_origin FROM HanYueDailyAction3D_copy



