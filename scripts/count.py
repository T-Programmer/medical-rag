from pymilvus import connections, Collection, utility

def count_entities():
    # 配置信息来源于 src/MedicalRag/config/app_config.yaml
    uri = "http://localhost:19530"
    token = "root:Milvus"
    collection_name = "medical_knowledge"

    print(f"Connecting to Milvus at {uri}...")
    try:
        connections.connect(uri=uri, token=token)
        print("Connected.")
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    if not utility.has_collection(collection_name):
        print(f"Collection '{collection_name}' does not exist.")
        return

    try:
        collection = Collection(collection_name)
        # 刷新以确保所有插入的数据都可见
        collection.flush() 
        count = collection.num_entities
        print(f"Collection '{collection_name}' has {count} entities.")
    except Exception as e:
        print(f"Error accessing collection: {e}")

if __name__ == "__main__":
    count_entities()