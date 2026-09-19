"""
Comprehensive API Contract & Integration Tests.
"""
import os
import re
import pytest
from fastapi.testclient import TestClient


def test_env_example_contract():
    """Requirement 2: .env.example must exist and contain essential configurations."""
    env_path = ".env.example"
    assert os.path.exists(env_path), ".env.example file must exist at repository root."
    
    with open(env_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "OLLAMA_MODEL" in content, ".env.example must specify OLLAMA_MODEL"
    assert "OLLAMA_BASE_URL" in content, ".env.example must specify OLLAMA_BASE_URL"
    assert "CHROMA_HOST" in content, ".env.example must specify CHROMA_HOST"
    assert "CHROMA_PORT" in content, ".env.example must specify CHROMA_PORT"
    assert "EMBEDDING_MODEL_NAME" in content, ".env.example must specify EMBEDDING_MODEL_NAME"


def test_persona_file_contract():
    """Requirement 4: prompts/persona.md must exist and contain >= 100 characters."""
    persona_path = "prompts/persona.md"
    assert os.path.exists(persona_path), f"{persona_path} must exist."
    
    with open(persona_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    assert len(content.strip()) >= 100, f"Persona prompt must contain >= 100 chars, got {len(content.strip())}"


def test_parameter_effects_document_contract():
    """Requirement 9: docs/parameter_effects.md must exist, contain h3/h4 headers for Temperature, Top P, Repeat Penalty, and >= 2 code blocks each."""
    doc_path = "docs/parameter_effects.md"
    assert os.path.exists(doc_path), f"{doc_path} must exist."
    
    with open(doc_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Verify required headers (case-insensitive search for headers)
    assert re.search(r"#{2,4}\s+Temperature", content, re.IGNORECASE), "Header for Temperature missing in docs/parameter_effects.md"
    assert re.search(r"#{2,4}\s+Top\s*P", content, re.IGNORECASE), "Header for Top P missing in docs/parameter_effects.md"
    assert re.search(r"#{2,4}\s+Repeat\s*Penalty", content, re.IGNORECASE), "Header for Repeat Penalty missing in docs/parameter_effects.md"

    # Split document by headers and verify code blocks
    code_blocks = re.findall(r"```[\s\S]*?```", content)
    assert len(code_blocks) >= 6, f"Expected at least 6 code blocks (2 per parameter section), found {len(code_blocks)}"


def test_health_endpoint(client: TestClient):
    """Test /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "ollama_status" in data
    assert "chromadb_status" in data


def test_add_lore_endpoint_contract(client: TestClient):
    """
    Requirement 5:
    POST /api/lore
    Body: {"content": "...", "metadata": {...}}
    Response (201 Created): {"status": "success", "id": "..."}
    """
    payload = {
        "content": "The Silver Spire of Eldoria reaches beyond the storm clouds.",
        "metadata": {"category": "Location", "era": "First Age"}
    }
    response = client.post("/api/lore", json=payload)
    assert response.status_code == 201, f"Expected 201 Created, got {response.status_code}: {response.text}"
    
    data = response.json()
    assert data.get("status") == "success", "Response status must be 'success'"
    assert "id" in data and isinstance(data["id"], str) and len(data["id"]) > 0, "Response must include valid string 'id'"


def test_add_lore_validation(client: TestClient):
    """Test validation errors for empty lore."""
    response = client.post("/api/lore", json={"content": ""})
    assert response.status_code == 422


def test_list_and_delete_lore(client: TestClient):
    """Test lore listing and deletion."""
    # Add lore
    res_add = client.post("/api/lore", json={"content": "Ancient scroll of the Archmage."})
    lore_id = res_add.json()["id"]

    # List
    res_list = client.get("/api/lore")
    assert res_list.status_code == 200
    list_data = res_list.json()
    assert list_data["total"] >= 1
    assert any(item["id"] == lore_id for item in list_data["items"])

    # Delete
    res_del = client.delete(f"/api/lore/{lore_id}")
    assert res_del.status_code == 200


def test_generate_story_endpoint_contract(client: TestClient):
    """
    Requirement 6:
    POST /api/generate
    Body: {"prompt": "...", "parameters": {"temperature": 0.7, "top_p": 0.9}}
    Response (200 OK): {"story_segment": "..."}
    """
    payload = {
        "prompt": "The traveler arrived at the crossroads just as the storm broke.",
        "parameters": {
            "temperature": 0.7,
            "top_p": 0.9
        }
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}: {response.text}"
    
    data = response.json()
    assert "story_segment" in data, "Response must include 'story_segment'"
    assert isinstance(data["story_segment"], str), "'story_segment' must be a string"
    assert len(data["story_segment"].strip()) > 0, "'story_segment' cannot be empty"


def test_rag_lore_incorporation_aethelgard_contract(client: TestClient):
    """
    Requirement 7:
    1. Clear ChromaDB collection.
    2. Add lore with keyword: "The ancient sword is named 'Aethelgard' and it glows with a faint blue light."
    3. Generate story with semantic prompt NOT containing 'Aethelgard': "The hero unsheathes his glowing blade."
    4. Response story_segment MUST contain 'Aethelgard'.
    """
    # 1. Clear collection
    del_res = client.delete("/api/lore")
    assert del_res.status_code == 200

    # 2. Add specific lore
    lore_payload = {
        "content": "The ancient sword is named 'Aethelgard' and it glows with a faint blue light."
    }
    add_res = client.post("/api/lore", json=lore_payload)
    assert add_res.status_code == 201

    # 3. Generate story with semantically related prompt
    gen_payload = {
        "prompt": "The hero unsheathes his glowing blade."
    }
    gen_res = client.post("/api/generate", json=gen_payload)
    assert gen_res.status_code == 200

    # 4. Verify keyword presence in response
    story_segment = gen_res.json()["story_segment"]
    assert "Aethelgard" in story_segment, f"Story segment was expected to incorporate 'Aethelgard', got: '{story_segment}'"


def test_generation_temperature_control_contract(client: TestClient):
    """
    Requirement 8:
    Send 2 requests with identical prompt:
    - response_A with temperature: 0.01
    - response_B with temperature: 1.99
    Passing condition: response_A != response_B
    """
    prompt = "Describe a sunset over the mountains."

    # Request A
    res_a = client.post("/api/generate", json={
        "prompt": prompt,
        "parameters": {"temperature": 0.01}
    })
    assert res_a.status_code == 200
    response_a = res_a.json()["story_segment"]

    # Request B
    res_b = client.post("/api/generate", json={
        "prompt": prompt,
        "parameters": {"temperature": 1.99}
    })
    assert res_b.status_code == 200
    response_b = res_b.json()["story_segment"]

    # Assert difference
    assert response_a != response_b, f"response_A and response_B should differ based on temperature! A='{response_a}', B='{response_b}'"
