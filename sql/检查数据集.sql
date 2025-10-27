

select distinct type,fileName from HanYueDailyAction3D_copy order by  filename

select distinct type,fileName from HanYueDailyAction3D_norm70_origin order by filename
select distinct type,fileName from HanYueDailyAction3D_origin order by filename
select distinct type,fileName from HanYueDailyAction3D_norm70_filtered_len_s order by filename
select distinct type,fileName from HanYueDailyAction3D_norm70 order by filename

select distinct type,fileName from HanYueDailyAction3D_norm70_test20 order by filename
select distinct * from HanYueDailyAction3D_test20 order by filename ,frameOrder

select distinct type,fileName from MSRAction3D_norm60 order by filename
select distinct type,fileName from MSRAction3D_copy order by  filename

select distinct type,fileName from Florence3DActions_copy order by filename
select distinct type,fileName from Florence3DActions_norm20 order by  filename

select * from hanyueDailyAction3D_norm70_test20 order by filename, frameOrder


--≤È—Ø≤‚ ‘ºØ
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



