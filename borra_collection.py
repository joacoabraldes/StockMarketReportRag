import argparse
import chromadb

# Definimos los argumentos del script
parser = argparse.ArgumentParser(description="Elimina una colección de ChromaDB.")
parser.add_argument("--collection", required=True, help="Nombre de la colección a eliminar")
args = parser.parse_args()

# Conectamos al cliente y eliminamos la colección
client = chromadb.PersistentClient(path="./chroma_db")
client.delete_collection(args.collection)

print(f"Colección '{args.collection}' eliminada correctamente.")
