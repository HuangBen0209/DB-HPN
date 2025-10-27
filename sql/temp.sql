SELECT TOP 0 * INTO MSRAction3D_60_hb FROM MSRAction3D;
SELECT TOP 0 * INTO MSRAction3D_50_hb FROM MSRAction3D;
SELECT TOP 0 * INTO MSRAction3D_40_hb FROM MSRAction3D;

SELECT TOP 0 * INTO HanYueDailyAction3D_100_hb FROM HanYueDailyAction3D;
SELECT TOP 0 * INTO HanYueDailyAction3D_70_hb FROM HanYueDailyAction3D;
SELECT TOP 0 * INTO HanYueDailyAction3D_60_hb FROM HanYueDailyAction3D;
SELECT TOP 0 * INTO HanYueDailyAction3D_50_hb FROM HanYueDailyAction3D;

SELECT TOP 0 * INTO Florence3DActions_17_hb FROM Florence3DActions;
SELECT TOP 0 * INTO Florence3DActions_20_hb FROM Florence3DActions;
SELECT TOP 0 * INTO Florence3DActions_23_hb FROM Florence3DActions;

SELECT TOP 0 * INTO UTKinectAction3D_30_hb FROM UTKinectAction3D;
SELECT TOP 0 * INTO UTKinectAction3D_40_hb FROM UTKinectAction3D;
SELECT TOP 0 * INTO UTKinectAction3D_50_hb FROM UTKinectAction3D;
SELECT TOP 0 * INTO UTKinectAction3D_60_hb FROM UTKinectAction3D;


drop table if exists MSRAction3D_60_hb;
drop table if exists MSRAction3D_50_hb;
drop table if exists MSRAction3D_40_hb;
drop table if exists HanYueDailyAction3D_100_hb;
drop table if exists HanYueDailyAction3D_70_hb;
drop table if exists HanYueDailyAction3D_60_hb;
drop table if exists HanYueDailyAction3D_50_hb;
drop table if exists Florence3DActions_17_hb;
drop table if exists Florence3DActions_20_hb;
drop table if exists Florence3DActions_23_hb;
drop table if exists UTKinectAction3D_30_hb;
drop table if exists UTKinectAction3D_40_hb;
drop table if exists UTKinectAction3D_50_hb;
drop table if exists UTKinectAction3D_60_hb;
drop table if exists UTKinectAction3D_70_hb;


select count(distinct filename) from Florence3DActions_test20
select count(distinct filename) from HanYueDailyAction3D_norm70_test20
select count(distinct filename) from HanYueDailyAction3D_origin
select count(distinct filename) from MSRAction3D_test20
select count(distinct filename) from UTKinectAction3D_test20


select count(distinct filename) from Florence3DActions
select count(distinct filename) from HanYueDailyAction3D
select count(distinct filename) from MSRAction3D
SELECT COUNT(DISTINCT CONCAT(filename, ',', type)) FROM UTKinectAction3D;