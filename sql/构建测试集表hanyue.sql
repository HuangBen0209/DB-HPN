-- 构建一HanYueDailyAction3D的副本：HanYueDailyAction3D_copy
-- 如果存在先删除它
IF OBJECT_ID('dbo.HanYueDailyAction3D_copy', 'U') IS NOT NULL
    DROP TABLE dbo.HanYueDailyAction3D_copy;
select * into HanYueDailyAction3D_copy from HanYueDailyAction3D
--删除某些样本
delete from HanYueDailyAction3D_copy where actiontypename='坐着鼓掌-站起来-站着鼓掌-原地跳-挥手'
delete from HanYueDailyAction3D_copy where actiontypename='喝水-坐下来-站起来-拍身上的灰-走'


--单纯复制表结构
if OBJECT_ID('dbo.HanYueDailyAction3D_filtered_len_s', 'U') IS NOT NULL
    DROP TABLE dbo.HanYueDailyAction3D_filtered_len_s;
if OBJECT_ID('dbo.HanYueDailyAction3D_origin', 'U') IS NOT NULL
    DROP TABLE dbo.HanYueDailyAction3D_origin;
SELECT TOP 0 * INTO HanYueDailyAction3D_filtered_len_s FROM HanYueDailyAction3D_copy
SELECT TOP 0 * INTO HanYueDailyAction3D_origin FROM HanYueDailyAction3D_copy


SELECT TOP 0 * INTO HanYueDailyAction3D_100_copy FROM HanYueDailyAction3D_copy

select count(frameOrder) ,filename from HanYueDailyAction3D_100_copy
GROUP BY filename

--构建不同比例的测试集，FileName 空表
IF OBJECT_ID('dbo.HanYueExperimentTest', 'U') IS NOT NULL
    DROP TABLE [dbo].[HanYueExperimentTest]
GO
CREATE TABLE [dbo].[HanYueExperimentTest] (
    [type] INT NOT NULL,          -- 动作编码
	[actiontypename] NVARCHAR(50) NOT NULL, --动作类型
    [typenum] INT NOT NULL,                -- 样本数量
    [testSamples15] TEXT NULL,             -- 存储 15% 测试集的文件名列表
    [testSamples20] TEXT NULL,             -- 存储 20% 测试集的文件名列表
    [testSamples30] TEXT NULL              -- 存储 30% 测试集的文件名列表
)
ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO

--------------------生成测试集数据表
TRUNCATE TABLE [dbo].[HanYueExperimentTest];
-- 游标变量
DECLARE @type INT, @actiontypename NVARCHAR(100), @total INT;
DECLARE @test15 NVARCHAR(MAX), @test20 NVARCHAR(MAX), @test30 NVARCHAR(MAX);
-- 定义游标：获取所有动作类型及数量
DECLARE action_cursor CURSOR FOR
SELECT [type], [actiontypename], COUNT(DISTINCT [fileName]) AS totalCount
FROM [dbo].[HanYueDailyAction3D_copy]
GROUP BY [type], [actiontypename];
OPEN action_cursor;
FETCH NEXT FROM action_cursor INTO @type, @actiontypename, @total;
WHILE @@FETCH_STATUS = 0
BEGIN
    -- 生成一个临时表来保存随机顺序的文件名
    IF OBJECT_ID('tempdb..#tempFiles') IS NOT NULL DROP TABLE #tempFiles;
    SELECT TOP (@total)
        ROW_NUMBER() OVER (ORDER BY NEWID()) AS rn,
        fileName
    INTO #tempFiles
    FROM [dbo].[HanYueDailyAction3D_copy]
    WHERE [type] = @type AND [actiontypename] = @actiontypename
    GROUP BY fileName;
    -- 前 15%
    SELECT @test15 = ISNULL(STUFF((
        SELECT ',' + QUOTENAME(fileName, '''')
        FROM #tempFiles WHERE rn <= CEILING(@total * 0.15)
        FOR XML PATH(''), TYPE).value('.', 'NVARCHAR(MAX)'), 1, 1, ''), '');
    -- 前 20%
    SELECT @test20 = ISNULL(STUFF((
        SELECT ',' + QUOTENAME(fileName, '''')
        FROM #tempFiles WHERE rn <= CEILING(@total * 0.20)
        FOR XML PATH(''), TYPE).value('.', 'NVARCHAR(MAX)'), 1, 1, ''), '');
    -- 前 30%
    SELECT @test30 = ISNULL(STUFF((
        SELECT ',' + QUOTENAME(fileName, '''')
        FROM #tempFiles WHERE rn <= CEILING(@total * 0.30)
        FOR XML PATH(''), TYPE).value('.', 'NVARCHAR(MAX)'), 1, 1, ''), '');
    -- 插入结果
    INSERT INTO [dbo].[HanYueExperimentTest] ([type], [actiontypename], [typenum], [testSamples15], [testSamples20], [testSamples30])
    VALUES (@type, @actiontypename, @total, @test15, @test20, @test30);
    -- 清空变量
    SET @test15 = NULL;
    SET @test20 = NULL;
    SET @test30 = NULL;
    FETCH NEXT FROM action_cursor INTO @type, @actiontypename, @total;
END;
CLOSE action_cursor;
DEALLOCATE action_cursor;

---添加testSamples0列
ALTER TABLE [dbo].[HanYueExperimentTest]
ADD [testSamples0] TEXT NULL;


-- 创建20%比例的测试集
IF OBJECT_ID('dbo.hanyueDailyAction3D_test20', 'U') IS NOT NULL
    DROP TABLE dbo.hanyueDailyAction3D_test20;
SELECT h.*
INTO hanyueDailyAction3D_test20
FROM hanyueDailyaction3D_copy h
JOIN (
   SELECT
    TRIM(REPLACE(REPLACE(REPLACE(REPLACE(value, '''', ''), CHAR(13), ''), CHAR(10), ''), '"', '')) AS FileName
FROM HanYueExperimentTest
CROSS APPLY STRING_SPLIT(CAST(testSamples20 AS NVARCHAR(MAX)), ',')
) AS t
ON h.FileName = t.FileName
WHERE t.FileName <> '';


