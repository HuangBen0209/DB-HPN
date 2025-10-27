-------------创建副本表------------
--创建对应的训练集表
if object_id('UTKinectAction3D_60_copy','U') is not null
    drop table UTKinectAction3D_60_copy;
--修改 tablekey 字段为 NVARCHAR(50)
-- 重新构建目标表并控制字段类型
SELECT
    CAST(tablekey AS NVARCHAR(50)) AS tablekey,  -- 转换为 NVARCHAR
    [type],
    [actor],
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
INTO UTKinectAction3D_60_copy
FROM UTKinectAction3D_60;

if object_id('UTKinectAction3D_60_origin','U') is not null
    drop table UTKinectAction3D_60_origin;
if object_id('UTKinectAction3D_60_filtered_len_s','U') is not null
    drop table UTKinectAction3D_60_filtered_len_s;
SELECT TOP 0 * INTO UTKinectAction3D_60_origin FROM UTKinectAction3D_60_copy;
SELECT TOP 0 * INTO UTKinectAction3D_60_filtered_len_s FROM UTKinectAction3D_60_copy;

if object_id('UT3DExperiment_60_Test','U') is not null
    drop table UT3DExperiment_60_Test;
-- 创建新表（使用 NVARCHAR(MAX) 替代 TEXT）
CREATE TABLE [dbo].[UT3DExperiment_60_Test] (
    [type] nvarchar(50) NOT NULL,                 -- 动作编码
    [typenum] INT NOT NULL,             -- 样本数量
    [testSamples0] NVARCHAR(MAX) NULL,  -- 存储 0% 测试集的文件名列表
    [testSamples15] NVARCHAR(MAX) NULL, -- 存储 15% 测试集的文件名列表
    [testSamples20] NVARCHAR(MAX) NULL, -- 存储 20% 测试集的文件名列表
    [testSamples30] NVARCHAR(MAX) NULL  -- 存储 30% 测试集的文件名列表
);
GO
----创建各比例测试集表
-- 清空测试表数据
TRUNCATE TABLE [dbo].[UT3DExperiment_60_Test];
-- 声明变量
DECLARE @type NVARCHAR(50);      -- 修改为 NVARCHAR 类型，防止 'a16' 转换错误
DECLARE @total INT;
DECLARE @test0 NVARCHAR(MAX), @test15 NVARCHAR(MAX), @test20 NVARCHAR(MAX), @test30 NVARCHAR(MAX);
-- 定义游标，获取每种动作类型及其对应样本数量
DECLARE action_cursor CURSOR FOR
SELECT [type], COUNT(DISTINCT [fileName]) AS totalCount
FROM [dbo].[UTKinectAction3D_60_copy]
GROUP BY [type];
-- 打开游标
OPEN action_cursor;
FETCH NEXT FROM action_cursor INTO @type, @total;
-- 遍历每一类动作
WHILE @@FETCH_STATUS = 0
BEGIN
    IF OBJECT_ID('tempdb..#tempFiles') IS NOT NULL DROP TABLE #tempFiles;
    SELECT TOP (@total)
        ROW_NUMBER() OVER (ORDER BY NEWID()) AS rn,
        fileName
    INTO #tempFiles
    FROM [dbo].[UTKinectAction3D_60_copy]
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
    INSERT INTO [dbo].[UT3DExperiment_60_Test]
        ([type], [typenum], [testSamples15], [testSamples20], [testSamples30], [testSamples0])
    VALUES
        (@type, @total, @test15, @test20, @test30, @test0);
    -- 清空变量
    SET @test0 = NULL;
    SET @test15 = NULL;
    SET @test20 = NULL;
    SET @test30 = NULL;
    FETCH NEXT FROM action_cursor INTO @type, @total;
END;
-- 关闭并释放游标
CLOSE action_cursor;
DEALLOCATE action_cursor;

-- 创建测试集20%
IF OBJECT_ID('dbo.UTKinectAction3D_60_test20', 'U') IS NOT NULL
    DROP TABLE dbo.UTKinectAction3D_60_test20;
SELECT h.*
INTO UTKinectAction3D_60_test20
FROM UTKinectAction3D_60_copy h
JOIN (
   SELECT
    TRIM(REPLACE(REPLACE(REPLACE(REPLACE(value, '''', ''), CHAR(13), ''), CHAR(10), ''), '"', '')) AS FileName
FROM UT3DExperiment_60_Test
CROSS APPLY STRING_SPLIT(CAST(testSamples20 AS NVARCHAR(MAX)), ',')

) AS t
ON h.FileName = t.FileName
WHERE t.FileName <> '';


select distinct fileName from UTKinectAction3D_60_copy
select distinct fileName from UTKinectAction3D_60_test20
