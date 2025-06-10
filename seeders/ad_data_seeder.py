"""
Active Directory data seeder.
"""
from seeders.chromadb_seeder import init_chromadb

def seed_ad_data():
    """
    Seed Active Directory data.
    """
    # This is a placeholder for AD data seeding functionality
    print("Seeding AD data...")
    
    # For now, just call the ChromaDB seeder
    init_chromadb()
    
    print("AD data seeding complete")

if __name__ == "__main__":
    seed_ad_data()
