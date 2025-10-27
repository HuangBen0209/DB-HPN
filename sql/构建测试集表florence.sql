--设置数据库兼容性
SELECT compatibility_level
FROM sys.databases
WHERE name = 'skeletonDataSet';
ALTER DATABASE skeletonDataSet SET COMPATIBILITY_LEVEL = 160;


if OBJECT_ID('dbo.Florence3DExperimentTest', 'U') IS NOT NULL
    DROP TABLE [dbo].[Florence3DExperimentTest]
if OBJECT_ID('Florence3DActions_copy', 'U') IS NOT NULL
    DROP TABLE Florence3DActions_copy;
if OBJECT_ID('Florence3DActions_origin', 'U') IS NOT NULL
    DROP TABLE Florence3DActions_origin;
if OBJECT_ID('Florence3DActions_filtered_len_s', 'U') IS NOT NULL
    DROP TABLE Florence3DActions_filtered_len_s;
if OBJECT_ID('Florence3DActions_test20', 'U') IS NOT NULL
    DROP TABLE Florence3DActions_test20;

--创建副本
select * into  Florence3DActions_copy from Florence3DActions
--复制表结构
SELECT TOP 0 * INTO Florence3DActions_origin FROM Florence3DActions_copy;
SELECT TOP 0 * INTO Florence3DActions_filtered_len_s FROM Florence3DActions_copy;

-- 创建不同测试集比例的表
CREATE TABLE [dbo].[Florence3DExperimentTest] (
    [type] INT NOT NULL,                 -- 动作编码
    [typenum] INT NOT NULL,             -- 样本数量
    [testSamples0] NVARCHAR(MAX) NULL,  -- 存储 0% 测试集的文件名列表
    [testSamples15] NVARCHAR(MAX) NULL, -- 存储 15% 测试集的文件名列表
    [testSamples20] NVARCHAR(MAX) NULL, -- 存储 20% 测试集的文件名列表
    [testSamples30] NVARCHAR(MAX) NULL  -- 存储 30% 测试集的文件名列表
);

--------------------生成测试集数据表
TRUNCATE TABLE [dbo].Florence3DExperimentTest;
-- 游标变量
DECLARE @type INT, @total INT;
DECLARE @test0 NVARCHAR(MAX), @test15 NVARCHAR(MAX), @test20 NVARCHAR(MAX), @test30 NVARCHAR(MAX);
-- 定义游标：获取所有动作类型及数量
DECLARE action_cursor CURSOR FOR
SELECT [type], COUNT(DISTINCT [fileName]) AS totalCount
FROM [dbo].[Florence3DActions_copy]
GROUP BY [type];
OPEN action_cursor;
FETCH NEXT FROM action_cursor INTO @type,  @total;
WHILE @@FETCH_STATUS = 0
BEGIN
    -- 生成一个临时表来保存随机顺序的文件名
    IF OBJECT_ID('tempdb..#tempFiles') IS NOT NULL DROP TABLE #tempFiles;
    SELECT TOP (@total)
        ROW_NUMBER() OVER (ORDER BY NEWID()) AS rn,
        fileName
    INTO #tempFiles
    FROM [dbo].[Florence3DActions_copy]
    WHERE [type] = @type
    GROUP BY fileName;
    -- 拼接前 0%
    SELECT @test0 = ISNULL(STUFF((
        SELECT ',' + QUOTENAME(fileName, '''')
        FROM #tempFiles
        WHERE rn <= 0  -- 抽样 0%
        FOR XML PATH(''), TYPE).value('.', 'NVARCHAR(MAX)'), 1, 1, ''), '');
    -- 拼接前 15%
    SELECT @test15 = ISNULL(STUFF((
        SELECT ',' + QUOTENAME(fileName, '''')
        FROM #tempFiles
        WHERE rn <= CEILING(@total * 0.15)
        FOR XML PATH(''), TYPE).value('.', 'NVARCHAR(MAX)'), 1, 1, ''), '');
    -- 拼接前 20%
    SELECT @test20 = ISNULL(STUFF((
        SELECT ',' + QUOTENAME(fileName, '''')
        FROM #tempFiles
        WHERE rn <= CEILING(@total * 0.20)
        FOR XML PATH(''), TYPE).value('.', 'NVARCHAR(MAX)'), 1, 1, ''), '');
    -- 拼接前 30%
    SELECT @test30 = ISNULL(STUFF((
        SELECT ',' + QUOTENAME(fileName, '''')
        FROM #tempFiles
        WHERE rn <= CEILING(@total * 0.30)
        FOR XML PATH(''), TYPE).value('.', 'NVARCHAR(MAX)'), 1, 1, ''), '');
    -- 插入结果
    INSERT INTO [dbo].[Florence3DExperimentTest]
        ([type], [typenum], [testSamples15], [testSamples20], [testSamples30], [testSamples0])
    VALUES
        (@type, @total, @test15, @test20, @test30, @test0);
    -- 清空变量
    SET @test0 = NULL;
    SET @test15 = NULL;
    SET @test20 = NULL;
    SET @test30 = NULL;
    -- 下一条记录
    FETCH NEXT FROM action_cursor INTO @type, @total;
END;
-- 释放游标
CLOSE action_cursor;
DEALLOCATE action_cursor;


-- 创建测试集20%
IF OBJECT_ID('dbo.Florence3DActions_test20', 'U') IS NOT NULL
    DROP TABLE dbo.Florence3DActions_test20;
SELECT h.*
INTO Florence3DActions_test20
FROM Florence3DActions_copy h
JOIN (
   SELECT
    TRIM(REPLACE(REPLACE(REPLACE(REPLACE(value, '''', ''), CHAR(13), ''), CHAR(10), ''), '"', '')) AS FileName
FROM Florence3DExperimentTest
CROSS APPLY STRING_SPLIT(CAST(testSamples20 AS NVARCHAR(MAX)), ',')
) AS t
ON h.FileName = t.FileName
WHERE t.FileName <> '';

select type, testSamples20 from Florence3DExperimentTest;

SELECT
    type,
    COUNT(distinct FileName) AS FileNameCount
FROM
    Florence3DActions_copy
GROUP BY
    type;

SELECT
    type,
    COUNT(distinct FileName) AS FileNameCount
FROM
    Florence3DActions_origin
GROUP BY
    type;