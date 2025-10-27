-- 查询每个FileName的最大frameorder，并按最大值从大到小排序
SELECT 
    FileName,
    MAX(frameorder) AS MaxFrameOrder  -- 取每个FileName对应的最大frameorder
FROM 
    UTKinectAction3D  -- 替换为你的表名
GROUP BY 
    FileName ,type -- 按FileName分组
ORDER BY 
    MaxFrameOrder DESC;  -- 按最大frameorder从大到小排序




    -- 查询每个FileName的最大frameorder，并按最大值从大到小排序
SELECT
    FileName,
    MAX(frameorder) AS MaxFrameOrder  -- 取每个FileName对应的最大frameorder
FROM
    HanYueDailyAction3D_copy  -- 替换为你的表名
GROUP BY
    FileName ,type -- 按FileName分组
ORDER BY
    MaxFrameOrder DESC;  -- 按最大frameorder从大到小排序



