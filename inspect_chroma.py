import chromadb

# Conectar al cliente
client = chromadb.PersistentClient(path="./chroma_db")

# Listar todas las colecciones
collections = client.list_collections()

if not collections:
    print("No hay colecciones en la base de datos.")
else:
    total_docs = 0
    print(f"Se encontraron {len(collections)} colecciones:\n")

    for c in collections:
        try:
            col = client.get_collection(c.name)
            count = col.count()
        except Exception as e:
            count = f"Error ({e})"
        print(f"- {c.name}: {count} documentos")
        if isinstance(count, int):
            total_docs += count

    print(f"\nTotal de documentos en todas las colecciones: {total_docs}")
