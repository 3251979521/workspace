# 1. Neo4j准备
## 1.1 建立实体约束
目的：视频中可能会出现多次同一个实体，为了使得同一个实体不被多次创建，明确规定所有标签为Entity的name属性相同的实体不可以多次被创建。
`CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE;`
## 1.2 撰写异步图谱管理器
文件名为`tkg_manager.py`，封装`TemporalKGManager`类，用于接收 VLM 吐出的三元组和时间戳，以异步非阻塞的方式将其转化为图谱中的事件节点。