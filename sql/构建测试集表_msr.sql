----------创建副本-----------------
if object_id('MSRAction3D_copy','U') is not null
    drop table MSRAction3D_copy;
--修改 tablekey 字段为 NVARCHAR(50)（或你想要的长度）
SELECT
    CAST(tablekey AS NVARCHAR(50)) AS tablekey,  -- 转换为 NVARCHAR
    [type],
    [fileName],
    [frameOrder],
    [spineBase],
    [spineMid],
    [neck],
    [head],
    [shoulderLeft],
    [ElbowLeft],
    [WristLeft],
    [HandLeft],
    [ShoulderRight],
    [ElbowRight],
    [WristRight],
    [HandRight],
    [HipLeft],
    [KneeLeft],
    [AnkleLeft],
    [FootLeft],
    [HipRight],
    [KneeRight],
    [AnkleRight],
    [FootRight],
    [SpineShoulder],
    [HandTipLeft],
    [ThumbLeft],
    [HandTipRight],
    [ThumbRight]
INTO MSRAction3D_copy
FROM msraction3D;

if object_id('MSRAction3D_origin','U') is not null
    drop table MSRAction3D_origin;
if object_id('MSRAction3D_filtered_len_s','U') is not null
    drop table MSRAction3D_filtered_len_s;
SELECT TOP 0 * INTO msraction3D_origin FROM MSRAction3D_copy;
SELECT TOP 0 * INTO MSRAction3D_filtered_len_s FROM MSRAction3D_copy;

---------- 创建测试集表
if object_id('msr3DExperimentTest','U') is not null
    drop table msr3DExperimentTest;
CREATE TABLE [dbo].[msr3DExperimentTest] (
    [type] nvarchar(50) NOT NULL,                 -- 动作编码
    [typenum] INT NOT NULL,             -- 样本数量
    [testSamples0] NVARCHAR(MAX) NULL,  -- 存储 0% 测试集的文件名列表
    [testSamples15] NVARCHAR(MAX) NULL, -- 存储 15% 测试集的文件名列表
    [testSamples20] NVARCHAR(MAX) NULL, -- 存储 20% 测试集的文件名列表
    [testSamples30] NVARCHAR(MAX) NULL  -- 存储 30% 测试集的文件名列表
);
GO

----创建各比例测试集表
TRUNCATE TABLE [dbo].[msr3DExperimentTest];
-- 声明变量
DECLARE @type NVARCHAR(50);
DECLARE @total INT;
DECLARE @test0 NVARCHAR(MAX), @test15 NVARCHAR(MAX), @test20 NVARCHAR(MAX), @test30 NVARCHAR(MAX);
DECLARE action_cursor CURSOR FOR
SELECT [type], COUNT(DISTINCT [fileName]) AS totalCount
FROM [dbo].[MSRAction3D_copy]
GROUP BY [type];
OPEN action_cursor;
FETCH NEXT FROM action_cursor INTO @type, @total;
WHILE @@FETCH_STATUS = 0
BEGIN
    IF OBJECT_ID('tempdb..#tempFiles') IS NOT NULL DROP TABLE #tempFiles;
    SELECT TOP (@total)
        ROW_NUMBER() OVER (ORDER BY NEWID()) AS rn,
        fileName
    INTO #tempFiles
    FROM [dbo].[msraction3D]
    WHERE [type] = @type
    GROUP BY fileName;
    -- 拼接前 0%
    SELECT @test0 = ISNULL(STUFF((
        SELECT ',' + QUOTENAME(fileName, '''')
        FROM #tempFiles
        WHERE rn <= 0
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

    -- 插入结果到测试信息表
    INSERT INTO [dbo].[msr3DExperimentTest]
        ([type], [typenum], [testSamples15], [testSamples20], [testSamples30], [testSamples0])
    VALUES
        (@type, @total, @test15, @test20, @test30, @test0);

    -- 清空变量
    SET @test0 = NULL;
    SET @test15 = NULL;
    SET @test20 = NULL;
    SET @test30 = NULL;

    -- 获取下一条记录
    FETCH NEXT FROM action_cursor INTO @type, @total;
END;

-- 关闭并释放游标
CLOSE action_cursor;
DEALLOCATE action_cursor;


-- 创建测试集20%
IF OBJECT_ID('dbo.MSRAction3D_test20', 'U') IS NOT NULL
    DROP TABLE dbo.MSRAction3D_test20;
SELECT h.*
INTO MSRAction3D_test20
FROM MSRAction3D_copy h
JOIN (
   SELECT
    TRIM(REPLACE(REPLACE(REPLACE(REPLACE(value, '''', ''), CHAR(13), ''), CHAR(10), ''), '"', '')) AS FileName
FROM msr3DExperimentTest
CROSS APPLY STRING_SPLIT(CAST(testSamples20 AS NVARCHAR(MAX)), ',')

) AS t
ON h.FileName = t.FileName
WHERE t.FileName <> '';

SELECT
    TYPE,
    COUNT (DISTINCT FILENAME)
FROM MSRAction3D
GROUP BY  TYPE

SELECT
    TYPE,
    COUNT (DISTINCT FILENAME)
FROM MSRAction3D_origin
GROUP BY  TYPE

SELECT
    TYPE,
    COUNT (DISTINCT FILENAME)
FROM MSRAction3D_filtered_len_s
GROUP BY  TYPE