# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import uuid
import os

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# --- OpenAPI Tool Imports ---
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset

# --- Check if API Key is set ---
if not os.environ.get('GOOGLE_API_KEY'):
    raise ValueError("GOOGLE_API_KEY environment variable is required. Please set it before running the script.")


# --- Constants ---
APP_NAME_OPENAPI = "openapi_petstore_app"
USER_ID_OPENAPI = "user_openapi_1"
SESSION_ID_OPENAPI = f"session_openapi_{uuid.uuid4()}"
AGENT_NAME_OPENAPI = "petstore_manager_agent"
GEMINI_MODEL = "gemini-2.0-flash"

# --- Sample OpenAPI Specification (JSON String) ---
# A basic Pet Store API example using httpbin.org as a mock server
openapi_spec_string = """
{
  "openapi": "3.0.0",
  "info": {
    "title": "Simple Pet Store API (Mock)",
    "version": "1.0.1",
    "description": "An API to manage pets in a store, using httpbin for responses."
  },
  "servers": [
    {
      "url": "https://httpbin.org",
      "description": "Mock server (httpbin.org)"
    }
  ],
  "paths": {
    "/get": {
      "get": {
        "summary": "List all pets (Simulated)",
        "operationId": "listPets",
        "description": "Simulates returning a list of pets. Uses httpbin's /get endpoint which echoes query parameters.",
        "parameters": [
          {
            "name": "limit",
            "in": "query",
            "description": "Maximum number of pets to return",
            "required": false,
            "schema": { "type": "integer", "format": "int32" }
          },
          {
             "name": "status",
             "in": "query",
             "description": "Filter pets by status",
             "required": false,
             "schema": { "type": "string", "enum": ["available", "pending", "sold"] }
          }
        ],
        "responses": {
          "200": {
            "description": "A list of pets (echoed query params).",
            "content": { "application/json": { "schema": { "type": "object" } } }
          }
        }
      }
    },
    "/post": {
      "post": {
        "summary": "Create a pet (Simulated)",
        "operationId": "createPet",
        "description": "Simulates adding a new pet. Uses httpbin's /post endpoint which echoes the request body.",
        "requestBody": {
          "description": "Pet object to add",
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "required": ["name"],
                "properties": {
                  "name": {"type": "string", "description": "Name of the pet"},
                  "tag": {"type": "string", "description": "Optional tag for the pet"}
                }
              }
            }
          }
        },
        "responses": {
          "201": {
            "description": "Pet created successfully (echoed request body).",
            "content": { "application/json": { "schema": { "type": "object" } } }
          }
        }
      }
    },
    "/get?petId={petId}": {
      "get": {
        "summary": "Info for a specific pet (Simulated)",
        "operationId": "showPetById",
        "description": "Simulates returning info for a pet ID. Uses httpbin's /get endpoint.",
        "parameters": [
          {
            "name": "petId",
            "in": "path",
            "description": "This is actually passed as a query param to httpbin /get",
            "required": true,
            "schema": { "type": "integer", "format": "int64" }
          }
        ],
        "responses": {
          "200": {
            "description": "Information about the pet (echoed query params)",
            "content": { "application/json": { "schema": { "type": "object" } } }
          },
          "404": { "description": "Pet not found (simulated)" }
        }
      }
    }
  }
}
"""

# --- Create OpenAPIToolset ---
petstore_toolset = OpenAPIToolset(
    spec_str=openapi_spec_string,
    spec_str_type='json',
)

# --- Agent Definition ---
root_agent = LlmAgent(
    name=AGENT_NAME_OPENAPI,
    model=GEMINI_MODEL,
    tools=[petstore_toolset],
    instruction="""Você é um assistente de Pet Store que gerencia pets via API.
    Use as ferramentas disponíveis para atender as solicitações do usuário.
    Responda em português de forma clara e amigável.
    Quando criar um pet, confirme os detalhes.
    Quando listar pets, mencione filtros usados.
    Quando mostrar um pet por ID, informe o ID solicitado.
    Seja conversacional e útil.
    """,
    description="Gerencia uma Pet Store usando ferramentas geradas de uma especificação OpenAPI."
)

# --- Session and Runner Setup ---
async def setup_session_and_runner():
    session_service_openapi = InMemorySessionService()
    runner_openapi = Runner(
        agent=root_agent,
        app_name=APP_NAME_OPENAPI,
        session_service=session_service_openapi,
    )
    await session_service_openapi.create_session(
        app_name=APP_NAME_OPENAPI,
        user_id=USER_ID_OPENAPI,
        session_id=SESSION_ID_OPENAPI,
    )
    return runner_openapi

# --- Agent Interaction Function ---
async def call_agent(query, runner_openapi):
    content = types.Content(role='user', parts=[types.Part(text=query)])
    final_response_text = "Agent não forneceu uma resposta final."
    
    try:
        async for event in runner_openapi.run_async(
            user_id=USER_ID_OPENAPI, 
            session_id=SESSION_ID_OPENAPI, 
            new_message=content
        ):
            if event.get_function_calls():
                call = event.get_function_calls()[0]
                print(f"🔧 Executando: {call.name}")
            elif event.is_final_response() and event.content and event.content.parts:
                final_response_text = event.content.parts[0].text.strip()
        
        return final_response_text
    except Exception as e:
        return f"Erro: {str(e)}"

# --- Interactive Terminal ---
async def interactive_terminal():
    print("🐕 ===== TERMINAL INTERATIVO - PET STORE AGENT =====")
    print("📋 Comandos disponíveis:")
    print("   - Listar pets: 'mostre os pets', 'quais pets estão disponíveis'")
    print("   - Criar pet: 'crie um gato chamado Mimi', 'adicione um cão Rex'")
    print("   - Buscar pet: 'mostre o pet 123', 'info do pet 456'")
    print("   - Sair: 'sair', 'quit', 'exit'")
    print("=" * 60)
    
    try:
        runner = await setup_session_and_runner()
        print("✅ Sistema inicializado com sucesso!")
        print("💬 Digite sua mensagem (ou 'sair' para terminar):")
        
        while True:
            try:
                user_input = input("\n🗣️  Você: ").strip()
                
                if user_input.lower() in ['sair', 'quit', 'exit', 'q']:
                    print("👋 Tchau! Até logo!")
                    break
                
                if not user_input:
                    print("⚠️  Por favor, digite algo!")
                    continue
                
                print("🤖 Agente: Processando...", end="", flush=True)
                response = await call_agent(user_input, runner)
                print(f"\r🤖 Agente: {response}")
                
            except KeyboardInterrupt:
                print("\n👋 Interrompido pelo usuário. Tchau!")
                break
            except Exception as e:
                print(f"\n❌ Erro: {e}")
                
    except Exception as e:
        print(f"❌ Erro ao inicializar o sistema: {e}")

# --- Execute ---
if __name__ == "__main__":
    try:
        asyncio.run(interactive_terminal())
    except KeyboardInterrupt:
        print("\n👋 Programa encerrado.")
    except Exception as e:
        print(f"❌ Erro fatal: {e}") 