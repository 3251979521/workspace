import asyncio
from neo4j import AsyncGraphDatabase

class TemporalKGManager:
    def __init__(self, uri, user, password):
        # 初始化异步驱动
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    async def close(self):
        await self.driver.close()

    async def insert_dynamic_event(self, subject_name, action_type, object_name, start_t, end_t, surprise_score=0.0):
        """
        异步插入时序事件节点及关系
        """
        cypher_query = """
        // 1. 确保主客体存在 (MERGE 避免重复创建)
        MERGE (sub:Entity {name: $sub_name})
        MERGE (obj:Entity {name: $obj_name})
        
        // 2. 创建独立的时间切片事件节点 (带有起始时间、结束时间和惊喜值)
        CREATE (evt:Event {
            type: $action_type, 
            start_time: $start_t, 
            end_time: $end_t,
            surprise_score: $surprise_score
        })
        
        // 3. 构建拓扑关系: Subject -> Event -> Object
        MERGE (sub)-[:ACTOR]->(evt)
        MERGE (evt)-[:TARGET]->(obj)
        
        RETURN evt.type AS action, evt.start_time AS start
        """
        
        # 开启异步会话执行写入
        async with self.driver.session() as session:
            result = await session.run(
                cypher_query, 
                sub_name=subject_name, 
                action_type=action_type, 
                obj_name=object_name, 
                start_t=start_t, 
                end_t=end_t,
                surprise_score=surprise_score
            )
            record = await result.single()
            print(f"[图谱写入成功] 动作: {record['action']}, 起始时间: {record['start']}s")

# ================= 单元测试 =================
async def main():
    kg = TemporalKGManager("bolt://localhost:7687", "neo4j", "zy159632")
    
    print("开始模拟异步并发写入 VLM 提取的 Chunk 数据...")
    
    # 模拟并发处理 3 个时间段的切片数据
    task1 = kg.insert_dynamic_event("Person_A", "Holding", "Cup_1", start_t=0.0, end_t=5.0, surprise_score=0.1)
    task2 = kg.insert_dynamic_event("Person_A", "Dropping", "Cup_1", start_t=5.0, end_t=5.5, surprise_score=0.92) # 高惊喜值突变！
    task3 = kg.insert_dynamic_event("Cup_1", "Shattered", "Floor", start_t=5.5, end_t=10.0, surprise_score=0.85)
    
    # 利用 asyncio 并发执行写入
    await asyncio.gather(task1, task2, task3)
    
    await kg.close()
    print("所有时序数据写入完毕！")

if __name__ == "__main__":
    asyncio.run(main())