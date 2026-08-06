"""
LoanDesk AI - Agent
The model chooses which tool to use: search company documents (RAG),
check live application status (Java Spring Boot API), or raise a disbursal
request (guarded by an approval limit).
"""

import sys

import anthropic
import requests
from rag_pipeline import collection   # reuse the vector store built earlier

# The model emits emoji; the default Windows console codepage (cp1252) cannot
# encode them and print() raises UnicodeEncodeError.
sys.stdout.reconfigure(encoding="utf-8")

llm = anthropic.Anthropic()

API_BASE = "http://localhost:8080/api/applications"
APPROVAL_LIMIT = 100000   # AED, from loan_policy.txt section 5
MAX_TOOL_TURNS = 5        # stop a runaway tool loop
REQUIRE_HUMAN_APPROVAL = True   # set False only for automated test runs


# ---------- TOOL IMPLEMENTATIONS ----------

def search_documents(query):
    """RAG tool: semantic search over policy, KYC and FAQ documents."""
    results = collection.query(query_texts=[query], n_results=3)
    context = ""
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context += f"[{meta['source']}]\n{doc}\n\n"
    return context


def check_application_status(app_id):
    """Integration tool: calls the Java Spring Boot service."""
    try:
        r = requests.get(f"{API_BASE}/{app_id}/status", timeout=5)
        if r.status_code == 404:
            return f"No application found with id {app_id}"
        return r.json()
    except requests.exceptions.ConnectionError:
        return "Application system is unreachable. Please check if the service is running."
    except requests.exceptions.Timeout:
        return "Application system timed out. Please retry."


def request_disbursal(app_id, amount, confirmed=False):
    if amount > APPROVAL_LIMIT:
        return (f"BLOCKED: AED {amount:,} exceeds the AED {APPROVAL_LIMIT:,} "
                f"auto-approval limit. Senior loan officer approval required. No action taken.")
    if not confirmed:
        return (f"CONFIRMATION REQUIRED: Disbursal of AED {amount:,} for application "
                f"{app_id} is within limits but needs operator confirmation. "
                f"Ask the user to confirm, then call this tool again with confirmed=true.")
    return f"Disbursal request created for application {app_id}, amount AED {amount:,}."


# ---------- TOOL DEFINITIONS: how the model sees the tools ----------

tools = [
    {
        "name": "search_documents",
        "description": (
            "Search company loan policy, KYC guidelines and customer FAQ documents. "
            "Use for any question about rules, eligibility, required documents, fees, "
            "processes or timelines."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "check_application_status",
        "description": (
            "Get the live status of a specific loan application from the core system. "
            "Use whenever the user mentions an application id or number."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "app_id": {"type": "string", "description": "The application id, e.g. 4521"}
            },
            "required": ["app_id"]
        }
    },
    {
        "name": "request_disbursal",
        "description": (
            "Raise a disbursal request for an approved application. "
            "High value disbursals are blocked and routed for human approval."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "app_id": {"type": "string"},
                "amount": {"type": "number", "description": "Amount in AED"},
                "confirmed": {
                    "type": "boolean",
                    "description": (
                        "Set to true only after the operator has explicitly "
                        "confirmed this specific disbursal."
                    )
                }
            },
            "required": ["app_id", "amount"]
        }
    }
]

def human_approves(app_id, amount):
    """Ask the operator directly, in the loop.

    This is the enforcement half of the disbursal gate. `confirmed` in the tool
    input is written by the *model*, so anything that can put text into its
    context — a user message, a document chunk, the `note` field the Java API
    returns — can set it. This prompt runs in Python, outside the model's reach,
    so a self-issued `confirmed: true` still has to clear a real keystroke.
    """
    if not REQUIRE_HUMAN_APPROVAL:
        return True
    try:
        reply = input(
            f"   >> APPROVE disbursal of AED {amount:,} "
            f"for application {app_id}? [y/N] "
        )
    except EOFError:
        # Non-interactive run: no operator to ask. Fail closed.
        print("   >> stdin closed, no operator to ask - denied")
        return False
    return reply.strip().lower() == "y"


TOOL_FUNCTIONS = {
    "search_documents": lambda i: search_documents(i["query"]),
    "check_application_status": lambda i: check_application_status(i["app_id"]),
    "request_disbursal": lambda i: request_disbursal(
        i["app_id"], i["amount"], i.get("confirmed", False)
    ),
}

SYSTEM_PROMPT = """You are LoanDesk AI, an assistant for loan operations staff.

Rules:
- For policy, eligibility or process questions, always use search_documents and cite
  the source file in brackets, for example [loan_policy.txt].
- For questions about a specific application, use check_application_status.
- If a question needs both live data and policy, use both tools.
- Never answer policy questions from your own knowledge. If the documents do not
  contain the answer, say you don't have that information in the company documents.
- When a disbursal is blocked by the approval limit, explain clearly why and what
  the user should do next.
- Distinguish between what the systems reported and what you inferred. Say "likely"
  when inferring."""


# ---------- AGENT LOOP ----------

def run_agent(user_message, history=None):
    messages = (history or []) + [{"role": "user", "content": user_message}]

    for _ in range(MAX_TOOL_TURNS):
        response = llm.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=800,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages
        )

        if response.stop_reason != "tool_use":
            # Model is done thinking, this is the final answer
            final = "".join(b.text for b in response.content if b.type == "text")
            if not final:
                # A refusal returns an empty content list, and max_tokens can
                # truncate before any text lands. Appending an empty assistant
                # message would make the *next* request 400, so bail instead.
                return f"[no answer returned: stop_reason={response.stop_reason}]", messages
            messages.append({"role": "assistant", "content": final})
            return final, messages

        # Model asked for one or more tools
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []

        for block in response.content:
            if block.type == "tool_use":
                print(f"   [tool: {block.name} {block.input}]")

                # Copy rather than mutate the SDK's block.
                tool_input = dict(block.input)

                # The model can set `confirmed` itself, so re-check with a human.
                if block.name == "request_disbursal" and tool_input.get("confirmed"):
                    if not human_approves(tool_input["app_id"], tool_input["amount"]):
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": ("DENIED: the operator declined this disbursal at "
                                        "the approval prompt. No action taken. Do not retry."),
                        })
                        continue

                # A raised exception here would kill the loop and lose history.
                # Hand the failure back to the model instead so it can recover.
                try:
                    result, is_error = TOOL_FUNCTIONS[block.name](tool_input), False
                except Exception as e:
                    result, is_error = f"Tool failed: {type(e).__name__}: {e}", True
                    print(f"   [tool error: {result}]")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result),
                    "is_error": is_error,
                })

        messages.append({"role": "user", "content": tool_results})

    return f"[stopped after {MAX_TOOL_TURNS} tool turns]", messages


# ---------- CHAT ----------

if __name__ == "__main__":
    print("LoanDesk Agent ready. Type 'quit' to exit.\n")
    history = []
    while True:
        q = input("You: ").strip()
        if q == "quit":
            break
        if not q:
            continue
        answer, history = run_agent(q, history)
        print(f"\nLoanDesk: {answer}\n")
