import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from azure.devops.connection import Connection
from msrest.authentication import BasicAuthentication
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import re
from typing import List, Dict, Tuple

load_dotenv()

# Configuration
AZURE_DEVOPS_ORG_URL = os.environ.get("AZURE_DEVOPS_ORG_URL")
PERSONAL_ACCESS_TOKEN = os.environ.get("AZURE_DEVOPS_PAT")
PROJECTS = ["EGPP"]
PERSIST_DIRECTORY = "chroma_db"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "all-mpnet-base-v2")

# Initialize Azure DevOps connection
if not PERSONAL_ACCESS_TOKEN:
    print("Error: Azure DevOps PAT not found. Ensure AZURE_DEVOPS_PAT is set in your .env file or environment variables.")
    exit(1)
if not AZURE_DEVOPS_ORG_URL:
    print("Error: Azure DevOps Org URL not found. Ensure AZURE_DEVOPS_ORG_URL is set in your .env file or environment variables.")
    exit(1)

credentials = BasicAuthentication("", PERSONAL_ACCESS_TOKEN)
connection = Connection(base_url=AZURE_DEVOPS_ORG_URL, creds=credentials)
work_item_client = connection.clients.get_work_item_tracking_client()

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

# Initialize embedding model and improved text splitter
try:
    model = SentenceTransformer(MODEL_PATH)
except Exception as e:
    print(f"Error loading SentenceTransformer model from {MODEL_PATH}: {e}")
    print("Ensure the model is downloaded and the MODEL_PATH is correct.")
    exit(1)

# Improved text splitter with better separators for work items
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80,
    separators=[
        "\n\n",  # Double newlines (paragraphs)
        "\n",    # Single newlines
        ". ",    # Sentence endings
        "! ",    # Exclamation sentences
        "? ",    # Question sentences
        "; ",    # Semicolons
        ", ",    # Commas
        " ",     # Spaces
        ""       # Character level (fallback)
    ],
    length_function=len,
    is_separator_regex=False,
)

def clean_and_normalize_text(text: str) -> str:
    """Clean and normalize text content for better embedding quality."""
    if not text:
        return ""
    
    # Remove HTML tags if any remain
    text = re.sub(r'<[^>]+>', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove excessive punctuation
    text = re.sub(r'[.]{3,}', '...', text)
    
    # Clean up common Azure DevOps artifacts
    text = re.sub(r'\[.*?\]', '', text)  # Remove bracketed references
    text = re.sub(r'&nbsp;|&amp;|&lt;|&gt;', ' ', text)  # HTML entities
    
    return text.strip()

def create_structured_chunks(work_item: Dict) -> List[Dict]:
    """Create semantically meaningful chunks with preserved context."""
    chunks = []
    
    # Clean the description
    description = clean_and_normalize_text(work_item.get('description', ''))
    title = clean_and_normalize_text(work_item.get('title', ''))
    
    # Create a comprehensive header for context
    header = (
        f"Work Item: {work_item.get('id', 'N/A')} "
        f"({work_item.get('type', 'N/A')}) "
        f"in {work_item.get('project', 'N/A')}"
    )
    
    # Strategy 1: Title + Description overview (always include)
    if title:
        overview_chunk = {
            'content': f"{header}\nTitle: {title}",
            'type': 'title',
            'priority': 'high'
        }
        
        # Add brief description if it fits
        if description and len(f"{overview_chunk['content']}\nSummary: {description[:200]}...") <= 400:
            overview_chunk['content'] += f"\nSummary: {description[:200]}..."
        
        chunks.append(overview_chunk)
    
    # Strategy 2: Detailed description chunks (if description exists and is substantial)
    if description and len(description) > 100:
        # Split description into semantic chunks
        desc_chunks = text_splitter.split_text(description)
        
        for i, chunk_text in enumerate(desc_chunks):
            # Add context header to each description chunk
            contextual_chunk = f"{header}\nTitle: {title}\nDescription Part {i+1}: {chunk_text}"
            
            chunks.append({
                'content': contextual_chunk,
                'type': 'description',
                'priority': 'medium',
                'part': i + 1,
                'total_parts': len(desc_chunks)
            })
    
    # Strategy 3: Metadata-rich chunk for search
    metadata_chunk = {
        'content': f"{header}\nTitle: {title}\nType: {work_item.get('type', 'N/A')}\nProject: {work_item.get('project', 'N/A')}",
        'type': 'metadata',
        'priority': 'high'
    }
    chunks.append(metadata_chunk)
    
    return chunks

def create_enhanced_metadata(work_item: Dict, chunk_info: Dict, chunk_index: int) -> Dict:
    """Create comprehensive metadata for better retrieval."""
    base_metadata = {
        "source": f"azure_devops_{work_item.get('project', 'unknown')}_wi_{work_item.get('id', 'unknown')}",
        "work_item_id": str(work_item.get("id", "unknown")),
        "project": work_item.get("project", "unknown"),
        "type": work_item.get("type", "unknown"),
        "title": work_item.get("title", "")[:100],  # Truncated title for metadata
        "chunk_type": chunk_info.get('type', 'unknown'),
        "chunk_priority": chunk_info.get('priority', 'medium'),
        "chunk_index": chunk_index,
        "has_description": bool(work_item.get('description', '').strip()),
        "content_length": len(chunk_info.get('content', ''))
    }
    
    # Add part information for description chunks
    if 'part' in chunk_info:
        base_metadata.update({
            "part_number": chunk_info['part'],
            "total_parts": chunk_info['total_parts']
        })
    
    return base_metadata

def fetch_work_items(project_name, custom_wiql_filter_clause=None, wiql_query_batch_limit=19000, details_batch_size=100):
    """
    Fetch work items from Azure DevOps for a given project in batches,
    paginating the WIQL query itself to avoid the 20k limit.
    Yields batches of work item details.
    """
    base_query_select = "Select [System.Id], [System.Title], [System.Description], [System.WorkItemType], [System.TeamProject] From WorkItems"
    base_project_filter = f"[System.TeamProject] = '{project_name}'"
    exclude_tasks_filter = "[System.WorkItemType] <> 'Task'"

    current_max_id = 0
    all_work_item_ids_fetched = []
    i = 0
    while i < 3:
        id_filter = f"[System.Id] > {current_max_id}"

        combined_base_filters = f"({base_project_filter}) AND ({id_filter}) AND ({exclude_tasks_filter})"

        if custom_wiql_filter_clause:
            query_where_clause = f"Where {combined_base_filters} AND ({custom_wiql_filter_clause})"
        else:
            query_where_clause = f"Where {combined_base_filters}"
            
        query = f"{base_query_select} {query_where_clause} Order By [System.Id]"
        wiql_object = {"query": query} 
        
        print(f"Executing paginated WIQL query for project '{project_name}' (ID > {current_max_id}, TOP {wiql_query_batch_limit}): {query[:250]}...")
        
        try:
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

def seed_project_collection(project_name, custom_wiql_filter_clause=None, 
                            wiql_query_batch_limit=19000, 
                            work_item_details_fetch_batch_size=100, 
                            processing_batch_size=200):
    """
    Seed ChromaDB collection for a specific project with Azure DevOps data,
    using improved chunking and embedding strategies.
    """
    print(f"Starting to seed collection for project: {project_name}")
    
    collection_name = f"{project_name}_shared"
    collection = client.get_or_create_collection(collection_name)
    
    total_chunks_added_for_project = 0
    
    for work_item_detail_batch in fetch_work_items(project_name, 
                                                   custom_wiql_filter_clause, 
                                                   wiql_query_batch_limit=wiql_query_batch_limit,
                                                   details_batch_size=work_item_details_fetch_batch_size):
        if not work_item_detail_batch:
            print(f"Skipping an empty or failed batch for project {project_name}.")
            continue

        print(f"Processing a batch of {len(work_item_detail_batch)} work item details for project {project_name}...")
        
        current_processing_batch_chunks = []
        current_processing_batch_metadatas = []
        current_processing_batch_ids = []

        for work_item in work_item_detail_batch:
            # Create structured chunks using the improved strategy
            structured_chunks = create_structured_chunks(work_item)
            
            for chunk_index, chunk_info in enumerate(structured_chunks):
                chunk_content = chunk_info['content']
                
                # Skip empty chunks
                if not chunk_content.strip():
                    continue
                
                current_processing_batch_chunks.append(chunk_content)
                
                # Create enhanced metadata
                metadata = create_enhanced_metadata(work_item, chunk_info, chunk_index)
                current_processing_batch_metadatas.append(metadata)
                
                # Create stable, unique IDs
                chunk_id = f"shared_chunk_{project_name}_wi_{work_item['id']}_{chunk_info['type']}_{chunk_index}"
                current_processing_batch_ids.append(chunk_id)

                # Process in batches
                if len(current_processing_batch_chunks) >= processing_batch_size:
                    try:
                        print(f"Generating embeddings for {len(current_processing_batch_chunks)} chunks...")
                        embeddings = model.encode(
                            current_processing_batch_chunks, 
                            show_progress_bar=False,
                            normalize_embeddings=True  # Normalize for better similarity computation
                        ).tolist()
                        
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
                        current_processing_batch_chunks = []
                        current_processing_batch_metadatas = []
                        current_processing_batch_ids = []
        
        # Add any remaining chunks
        if current_processing_batch_chunks:
            try:
                print(f"Generating embeddings for remaining {len(current_processing_batch_chunks)} chunks...")
                embeddings = model.encode(
                    current_processing_batch_chunks, 
                    show_progress_bar=False,
                    normalize_embeddings=True
                ).tolist()
                
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
    """Main function to seed ChromaDB with Azure DevOps data using improved chunking."""
    custom_project_filters = {}

    # Optimized batch sizes for better quality
    wiql_query_page_limit = 19000
    work_item_details_api_batch_size = 50
    db_processing_batch_size = 100

    try:
        for project in PROJECTS:
            print(f"Processing project: {project}")
            project_specific_filter = custom_project_filters.get(project)
            seed_project_collection(
                project, 
                custom_wiql_filter_clause=project_specific_filter,
                wiql_query_batch_limit=wiql_query_page_limit,
                work_item_details_fetch_batch_size=work_item_details_api_batch_size,
                processing_batch_size=db_processing_batch_size
            )
        print("Shared ChromaDB collections seeding process completed.")
    except Exception as e:
        print(f"An unexpected error occurred in the main seeding process: {str(e)}")

if __name__ == "__main__":
    main()