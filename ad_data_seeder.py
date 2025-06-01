import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from azure.devops.connection import Connection
from msrest.authentication import BasicAuthentication
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
# from bs4 import BeautifulSoup # Uncomment if you need to parse HTML from descriptions
load_dotenv()
# Configuration
# IMPORTANT: Storing secrets like PERSONAL_ACCESS_TOKEN directly in code is a security risk.
# Consider using environment variables or a secure vault.
AZURE_DEVOPS_ORG_URL = os.environ.get("AZURE_DEVOPS_ORG_URL") # Removed default for clarity with .env
PERSONAL_ACCESS_TOKEN = os.environ.get("AZURE_DEVOPS_PAT") # Removed default
PROJECTS = ["EGPP"]  # List of Azure DevOps projects
PERSIST_DIRECTORY = "chroma_db"
# Ensure the model path is correct and the model is present
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "all-mpnet-base-v2")


# Initialize Azure DevOps connection
if not PERSONAL_ACCESS_TOKEN: # Simplified check
    print("Error: Azure DevOps PAT not found. Ensure AZURE_DEVOPS_PAT is set in your .env file or environment variables.")
    exit(1)
if not AZURE_DEVOPS_ORG_URL: # Simplified check
    print("Error: Azure DevOps Org URL not found. Ensure AZURE_DEVOPS_ORG_URL is set in your .env file or environment variables.")
    exit(1)

credentials = BasicAuthentication("", PERSONAL_ACCESS_TOKEN)
connection = Connection(base_url=AZURE_DEVOPS_ORG_URL, creds=credentials)
work_item_client = connection.clients.get_work_item_tracking_client()

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

# Initialize embedding model and text splitter
try:
    model = SentenceTransformer(MODEL_PATH)
except Exception as e:
    print(f"Error loading SentenceTransformer model from {MODEL_PATH}: {e}")
    print("Ensure the model is downloaded and the MODEL_PATH is correct.")
    exit(1)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

def fetch_work_items(project_name, custom_wiql_filter_clause=None, wiql_query_batch_limit=19000, details_batch_size=100):
    """
    Fetch work items from Azure DevOps for a given project in batches,
    paginating the WIQL query itself to avoid the 20k limit.
    Yields batches of work item details.
    """
    base_query_select = "Select [System.Id], [System.Title], [System.Description], [System.WorkItemType], [System.TeamProject] From WorkItems"
    base_project_filter = f"[System.TeamProject] = '{project_name}'"
    exclude_tasks_filter = "[System.WorkItemType] <> 'Task'"

    current_max_id = 0 # Start with 0 to get all IDs initially
    all_work_item_ids_fetched = []
    i = 0
    while i < 3:
        id_filter = f"[System.Id] > {current_max_id}"

        combined_base_filters = f"({base_project_filter}) AND ({id_filter}) AND ({exclude_tasks_filter})"

        if custom_wiql_filter_clause:
            query_where_clause = f"Where {combined_base_filters} AND ({custom_wiql_filter_clause})"
        else:
            query_where_clause = f"Where {combined_base_filters}"
            
        # Always order by ID for pagination
        query = f"{base_query_select} {query_where_clause} Order By [System.Id]"
        
        # Prepare the Wiql object for the API call
        wiql_object = {"query": query} 
        
        print(f"Executing paginated WIQL query for project '{project_name}' (ID > {current_max_id}, TOP {wiql_query_batch_limit}): {query[:250]}...")
        
        try:
            # Use the 'top' parameter of the query_by_wiql method
            query_result = work_item_client.query_by_wiql(wiql_object, top=wiql_query_batch_limit)
            query_result_refs = query_result.work_items
        except Exception as e:
            print(f"Error executing paginated WIQL query for project {project_name}: {e}")
            break 

        if not query_result_refs:
            print(f"No more work item references found for project {project_name} (ID > {current_max_id}).")
            break

        batch_ids = [ref.id for ref in query_result_refs]
        all_work_item_ids_fetched.extend(batch_ids)
        
        if not batch_ids: 
            break

        current_max_id = max(batch_ids) 
        
        if len(batch_ids) < wiql_query_batch_limit:
            print(f"Fetched the last batch of {len(batch_ids)} work item references.")
            break
        else:
            print(f"Fetched a batch of {len(batch_ids)} work item references. Max ID in batch: {current_max_id}.")
    i+=1
    if not all_work_item_ids_fetched:
        print(f"No work items found for project {project_name} with the given criteria.")
        yield []
        return

    print(f"Total {len(all_work_item_ids_fetched)} work item references collected for project '{project_name}'. Fetching details in batches of {details_batch_size}...")
    
    fields_to_retrieve = [
        "System.Id", 
        "System.Title", 
        "System.Description", 
        "System.WorkItemType", 
        "System.TeamProject"
    ]

    for i in range(0, len(all_work_item_ids_fetched), details_batch_size):
        batch_ids_for_details = all_work_item_ids_fetched[i:i + details_batch_size]
        if not batch_ids_for_details:
            continue
        
        try:
            batch_work_items_details = work_item_client.get_work_items(
                ids=batch_ids_for_details, 
                fields=fields_to_retrieve,
                error_policy='omit'
            )
            
            processed_batch = []
            for work_item_detail in batch_work_items_details:
                description = work_item_detail.fields.get("System.Description", "")
                # if description: # Uncomment for HTML parsing
                #     from bs4 import BeautifulSoup
                #     soup = BeautifulSoup(description, "html.parser")
                #     description = soup.get_text(separator=" ")

                processed_batch.append({
                    "id": work_item_detail.id,
                    "title": work_item_detail.fields.get("System.Title", ""),
                    "description": description,
                    "type": work_item_detail.fields.get("System.WorkItemType", ""),
                    "project": work_item_detail.fields.get("System.TeamProject", project_name) 
                })
            if processed_batch:
                yield processed_batch
        except Exception as e:
            print(f"Error fetching details for work item batch in project {project_name} (IDs: {batch_ids_for_details[:5]}...): {e}")
            continue

def process_work_item(work_item):
    """Process work item into a single text string for chunking."""
    # Ensure description is handled if it's None or empty
    description_text = work_item.get('description', "") if work_item.get('description') else "No description provided."
    text = (
        f"Work Item ID: {work_item.get('id', 'N/A')}\n"
        f"Type: {work_item.get('type', 'N/A')}\n"
        f"Title: {work_item.get('title', 'N/A')}\n"
        f"Project: {work_item.get('project', 'N/A')}\n" # Added project to the text
        f"Description: {description_text}"
    )
    return text

def seed_project_collection(project_name, custom_wiql_filter_clause=None, 
                            wiql_query_batch_limit=19000, 
                            work_item_details_fetch_batch_size=100, 
                            processing_batch_size=200):
    """
    Seed ChromaDB collection for a specific project with Azure DevOps data,
    processing in batches for scalability.
    """
    print(f"Starting to seed collection for project: {project_name}")
    
    collection_name = f"{project_name}_shared"
    # Idempotency: get_or_create_collection ensures the collection exists.
    # Subsequent adds with IDs will update existing items (upsert).
    collection = client.get_or_create_collection(collection_name)
    
    total_chunks_added_for_project = 0
    
    # fetch_work_items is now a generator yielding batches of work items
    for work_item_detail_batch in fetch_work_items(project_name, 
                                                   custom_wiql_filter_clause, 
                                                   wiql_query_batch_limit=wiql_query_batch_limit, # Corrected parameter name
                                                   details_batch_size=work_item_details_fetch_batch_size):
        if not work_item_detail_batch:
            print(f"Skipping an empty or failed batch for project {project_name}.")
            continue

        print(f"Processing a batch of {len(work_item_detail_batch)} work item details for project {project_name}...")
        
        current_processing_batch_chunks = []
        current_processing_batch_metadatas = []
        current_processing_batch_ids = []

        for work_item in work_item_detail_batch:
            text_to_chunk = process_work_item(work_item)
            chunks = text_splitter.split_text(text_to_chunk)
            for i, chunk_content in enumerate(chunks):
                current_processing_batch_chunks.append(chunk_content)
                current_processing_batch_metadatas.append({
                    "source": f"azure_devops_{project_name}_wi_{work_item['id']}",
                    "work_item_id": str(work_item["id"]), # Ensure ID is string for metadata
                    "project": project_name,
                    "type": work_item["type"]
                })
                # ID for each chunk, ensuring uniqueness and stability for idempotency
                current_processing_batch_ids.append(f"shared_chunk_{project_name}_wi_{work_item['id']}_{i}")

                # If the current processing batch reaches its size, embed and add to DB
                if len(current_processing_batch_chunks) >= processing_batch_size:
                    try:
                        print(f"Generating embeddings for {len(current_processing_batch_chunks)} chunks...")
                        embeddings = model.encode(current_processing_batch_chunks, show_progress_bar=False).tolist()
                        collection.add(
                            documents=current_processing_batch_chunks,
                            embeddings=embeddings,
                            metadatas=current_processing_batch_metadatas,
                            ids=current_processing_batch_ids
                        )
                        total_chunks_added_for_project += len(current_processing_batch_chunks)
                        print(f"Added {len(current_processing_batch_chunks)} chunks to collection '{collection_name}'.")
                    except Exception as e:
                        print(f"Error processing or adding a sub-batch to ChromaDB for project {project_name}: {e}")
                    finally:
                        # Reset for the next processing batch
                        current_processing_batch_chunks = []
                        current_processing_batch_metadatas = []
                        current_processing_batch_ids = []
        
        # Add any remaining chunks from the last work_item_detail_batch
        if current_processing_batch_chunks:
            try:
                print(f"Generating embeddings for remaining {len(current_processing_batch_chunks)} chunks...")
                embeddings = model.encode(current_processing_batch_chunks, show_progress_bar=False).tolist()
                collection.add(
                    documents=current_processing_batch_chunks,
                    embeddings=embeddings,
                    metadatas=current_processing_batch_metadatas,
                    ids=current_processing_batch_ids
                )
                total_chunks_added_for_project += len(current_processing_batch_chunks)
                print(f"Added remaining {len(current_processing_batch_chunks)} chunks to collection '{collection_name}'.")
            except Exception as e:
                print(f"Error processing or adding the final sub-batch to ChromaDB for project {project_name}: {e}")
            
    if total_chunks_added_for_project > 0:
        print(f"Successfully seeded/updated collection '{collection_name}' for project '{project_name}' with a total of {total_chunks_added_for_project} chunks.")
    else:
        print(f"No new chunks were added to collection '{collection_name}' for project '{project_name}'. This might be normal if data hasn't changed or no items matched the query.")

def main():
    """Main function to seed ChromaDB with Azure DevOps data."""
    # Example of a custom WIQL filter clause.
    # This could be made more dynamic, e.g., read from a config file or command-line arguments.
    # Example: Only 'Bug' work items that are 'Active'
    # custom_project_filters = {
    #     "YourProjectName1": "[System.WorkItemType] = 'Bug' AND [System.State] = 'Active'",
    #     "YourProjectName2": "[System.WorkItemType] = 'User Story'"
    # }
    custom_project_filters = {} # No custom filters by default for all projects

    # Batch sizes
    # Consider lowering this if 19000 still causes issues, e.g., to 10000 or 15000
    wiql_query_page_limit = 19000  # How many work item references to fetch per WIQL query API call
    work_item_details_api_batch_size = 50
    db_processing_batch_size = 100

    try:
        for project in PROJECTS:
            print(f"Processing project: {project}")
            project_specific_filter = custom_project_filters.get(project)
            seed_project_collection(
                project, 
                custom_wiql_filter_clause=project_specific_filter,
                wiql_query_batch_limit=wiql_query_page_limit, # Pass the correct parameter
                work_item_details_fetch_batch_size=work_item_details_api_batch_size,
                processing_batch_size=db_processing_batch_size
            )
        print("Shared ChromaDB collections seeding process completed.")
    except Exception as e:
        print(f"An unexpected error occurred in the main seeding process: {str(e)}")

if __name__ == "__main__":
    main()