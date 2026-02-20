import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools import StructuredTool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

import logging

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool input schemas
# ---------------------------------------------------------------------------

class GetEmployeeInfoInput(BaseModel):
    employee_names: List[str] = Field(
        description="List of employee names to query. Use full names as they appear in the database."
    )
    fields: List[str] = Field(
        description=(
            "Fields to retrieve. Available options: "
            "salary, vacation_leave, sick_leave, service_incentive_leave, "
            "position, rank, hire_date, regularization_date, employment_status, organizational_unit"
        )
    )


class CalculateHRMetricsInput(BaseModel):
    employee_names: List[str] = Field(
        description="List of employee names to calculate for."
    )
    calculation_type: str = Field(
        description=(
            "Type of calculation: "
            "'leave_encashment' (calculates encashable leave payout), "
            "'salary_raise' (calculates new salary after raise), "
            "'benefits_cost' (calculates benefits cost based on rate), "
            "'overtime_pay' (calculates overtime pay)"
        )
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Additional parameters depending on calculation_type: "
            "For salary_raise: {'raise_percentage': 10} (number, not decimal). "
            "For benefits_cost: {'benefits_rate': 15} (number, not decimal). "
            "For overtime_pay: {'overtime_hours': 8}. "
            "For leave_encashment: no params needed."
        )
    )
    return_aggregate: bool = Field(
        default=False,
        description=(
            "Set to True when the user asks for a TOTAL or COMBINED amount across multiple employees. "
            "Returns both per-employee breakdown and a grand total."
        )
    )


class QueryPolicyInput(BaseModel):
    question: str = Field(
        description="The policy question to search for in HR policy documents."
    )


# ---------------------------------------------------------------------------
# Conversation state (per-user session)
# ---------------------------------------------------------------------------

class ConversationState:
    def __init__(self):
        self.messages: List[Dict[str, str]] = []
        self.summary: str = ""
        self.last_mentioned_employees: List[str] = []

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def get_last_n(self, n: int = 3) -> List[Dict[str, str]]:
        return self.messages[-n:] if len(self.messages) >= n else self.messages

    def should_summarize(self) -> bool:
        return len(self.messages) >= 10

    def update_last_mentioned(self, names: List[str]):
        for name in names:
            if name and name not in self.last_mentioned_employees:
                self.last_mentioned_employees.insert(0, name)
        self.last_mentioned_employees = self.last_mentioned_employees[:5]


# ---------------------------------------------------------------------------
# User authentication
# ---------------------------------------------------------------------------

class UserAuth:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.current_user: Optional[str] = None
        self.current_role: Optional[str] = None
        self.is_authenticated: bool = False
        self._hr_admin = {"username": "hr_admin", "password": "hrpassword123", "role": "hr"}
        logger.info(f"UserAuth initialized with {len(df)} employees")

    def authenticate(self, username: str, password: str) -> Tuple[bool, str]:
        try:
            if username == self._hr_admin["username"] and password == self._hr_admin["password"]:
                self.current_user = username
                self.current_role = "hr"
                self.is_authenticated = True
                return True, "Welcome HR Admin!"

            username = str(username).strip()
            password = str(password).strip()
            self.df['employee_id'] = self.df['employee_id'].astype(str)
            user_row = self.df[self.df['employee_id'] == username]

            if user_row.empty:
                return False, "Invalid employee ID."

            stored_pw = str(user_row['password'].values[0]).strip()
            if stored_pw != password:
                return False, "Incorrect password."

            self.current_user = str(user_row['name'].values[0])
            position = str(user_row['position'].values[0]).lower()
            self.current_role = "hr" if "hr" in position else "employee"
            self.is_authenticated = True
            return True, f"Welcome {self.current_user}!"

        except Exception as e:
            logger.error(f"Authentication error: {e}", exc_info=True)
            return False, "Authentication error."

    def logout(self) -> str:
        name = self.current_user or "User"
        self.current_user = None
        self.current_role = None
        self.is_authenticated = False
        return f"Goodbye {name}!"


# ---------------------------------------------------------------------------
# HR Agent
# ---------------------------------------------------------------------------

FIELD_MAP = {
    "salary": "basic_pay_in_php",
    "vacation_leave": "vacation_leave",
    "sick_leave": "sick_leave",
    "service_incentive_leave": "service_incentive_leave",
    "position": "position",
    "rank": "rank",
    "hire_date": "hire_date",
    "regularization_date": "regularization_date",
    "employment_status": "employment_status",
    "organizational_unit": "organizational_unit",
}

SYSTEM_PROMPT = """You are an intelligent HR Assistant with access to employee data and company HR policies.

=== Session Context ===
Requester Role: {role}
Requester Name: {user_name}
Requester Employee ID: {employee_id}
Recently mentioned employees in this conversation: {last_mentioned_employees}

=== Conversation Summary ===
{conversation_summary}

=== Tool Usage Guidelines ===
- Use `get_employee_info` for: salary, leave balances, position, hire date, employment status queries.
- Use `calculate_hr_metrics` for: encashment amounts, salary raise calculations, benefits cost, overtime pay.
  → Set return_aggregate=True when the user asks for totals, combined costs, or grand totals across multiple people.
- Use `query_policy` for: ANY question about HR rules, procedures, eligibility, or policy details.
  → Always cite the policy source in your answer.

=== Important Rules ===
- When the user says "this person", "them", "same for them", "all of them" → refer to the recently mentioned employees list above.
- Always respond in the same language as the user's message.
- When presenting monetary values, use the ₱ symbol (Philippine Peso).
- If a calculation involves multiple employees, present a clean breakdown per person, then show the total if requested.
"""


class HRAgent:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        csv_path = os.path.join(base_dir, 'data', 'sample_employee.csv')
        self.df = pd.read_csv(csv_path)
        self.df['employee_id'] = self.df['employee_id'].astype(str)
        self.employee_names = self.df['name'].str.strip().tolist()

        self.auth = UserAuth(self.df)

        self.llm = ChatOpenAI(
            model_name=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        # Load ChromaDB vector store
        self.vectorstore = None
        try:
            from store_embeddings import create_embeddings
            self.vectorstore = create_embeddings()
        except Exception as e:
            self.logger.error(f"ChromaDB initialization failed: {e}")

        # Per-user session states
        self._sessions: Dict[str, ConversationState] = {}

        # Mutable context set before each request (tools read from this)
        self._current_user_context: Dict[str, Any] = {}

        self._build_agent()
        self.logger.info("HRAgent initialized successfully")

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _get_session(self, username: str) -> ConversationState:
        if username not in self._sessions:
            self._sessions[username] = ConversationState()
        return self._sessions[username]

    def clear_session(self, username: str):
        self._sessions.pop(username, None)

    # ------------------------------------------------------------------
    # Deterministic access control (runs inside every tool call)
    # ------------------------------------------------------------------

    def _validate_access(
        self,
        employee_names: List[str],
        requester_role: str,
        requester_name: str
    ) -> Tuple[bool, str]:
        if requester_role == "hr":
            return True, "Access granted"
        for name in employee_names:
            if name.strip().lower() != requester_name.strip().lower():
                return False, (
                    f"Access denied: As an employee, you can only view your own information. "
                    f"You do not have permission to access data for '{name}'."
                )
        return True, "Access granted"

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _tool_get_employee_info(self, employee_names: List[str], fields: List[str]) -> str:
        ctx = self._current_user_context
        ok, msg = self._validate_access(
            employee_names,
            ctx.get("role", "employee"),
            ctx.get("name", "")
        )
        if not ok:
            return json.dumps({"error": msg})

        session = self._get_session(ctx.get("username", ""))
        session.update_last_mentioned(employee_names)

        results = []
        for name in employee_names:
            match = self.df[self.df['name'].str.strip().str.lower() == name.strip().lower()]
            if match.empty:
                results.append({"employee": name, "error": "Employee not found in database."})
                continue

            row = match.iloc[0]
            data: Dict[str, Any] = {"employee": name}
            for field in fields:
                col = FIELD_MAP.get(field.lower())
                if col and col in self.df.columns:
                    val = row[col]
                    data[field] = int(val) if field in ("vacation_leave", "sick_leave", "service_incentive_leave") else val
                else:
                    data[field] = f"Field '{field}' is not available."
            results.append(data)

        return json.dumps(results, ensure_ascii=False, default=str)

    def _tool_calculate_hr_metrics(
        self,
        employee_names: List[str],
        calculation_type: str,
        params: Dict[str, Any] = {},
        return_aggregate: bool = False
    ) -> str:
        ctx = self._current_user_context
        ok, msg = self._validate_access(
            employee_names,
            ctx.get("role", "employee"),
            ctx.get("name", "")
        )
        if not ok:
            return json.dumps({"error": msg})

        session = self._get_session(ctx.get("username", ""))
        session.update_last_mentioned(employee_names)

        results = []
        grand_total = 0.0

        for name in employee_names:
            match = self.df[self.df['name'].str.strip().str.lower() == name.strip().lower()]
            if match.empty:
                results.append({"employee": name, "error": "Employee not found."})
                continue

            row = match.iloc[0]
            salary = float(row['basic_pay_in_php'])

            if calculation_type == "leave_encashment":
                vl = int(row.get('vacation_leave', 0))
                sil = int(row.get('service_incentive_leave', 0)) if 'service_incentive_leave' in row.index else 0
                daily_rate = salary / 30
                vl_amount = round(daily_rate * vl, 2)
                sil_amount = round(daily_rate * sil, 2)
                total = round(vl_amount + sil_amount, 2)
                grand_total += total
                results.append({
                    "employee": name,
                    "monthly_salary": salary,
                    "daily_rate": round(daily_rate, 2),
                    "vacation_leave_days": vl,
                    "vacation_leave_encashment": vl_amount,
                    "service_incentive_leave_days": sil,
                    "service_incentive_leave_encashment": sil_amount,
                    "total_encashment": total,
                    "currency": "PHP",
                    "note": "Sick Leave cannot be encashed per policy."
                })

            elif calculation_type == "salary_raise":
                pct = float(params.get("raise_percentage", 0))
                increase = round(salary * pct / 100, 2)
                new_salary = round(salary + increase, 2)
                grand_total += new_salary
                results.append({
                    "employee": name,
                    "current_salary": salary,
                    "raise_percentage": pct,
                    "salary_increase": increase,
                    "new_salary": new_salary,
                    "currency": "PHP"
                })

            elif calculation_type == "benefits_cost":
                rate = float(params.get("benefits_rate", 0))
                cost = round(salary * rate / 100, 2)
                total_comp = round(salary + cost, 2)
                grand_total += cost
                results.append({
                    "employee": name,
                    "base_salary": salary,
                    "benefits_rate_percent": rate,
                    "benefits_cost": cost,
                    "total_compensation": total_comp,
                    "currency": "PHP"
                })

            elif calculation_type == "overtime_pay":
                hours = float(params.get("overtime_hours", 0))
                # Hourly rate = monthly salary / (22 working days * 8 hours)
                hourly_rate = salary / (22 * 8)
                ot_pay = round(hourly_rate * 1.25 * hours, 2)
                grand_total += ot_pay
                results.append({
                    "employee": name,
                    "monthly_salary": salary,
                    "overtime_hours": hours,
                    "hourly_rate": round(hourly_rate, 2),
                    "overtime_premium": "25%",
                    "overtime_pay": ot_pay,
                    "currency": "PHP"
                })

            else:
                results.append({
                    "employee": name,
                    "error": f"Unknown calculation_type '{calculation_type}'. Use: leave_encashment, salary_raise, benefits_cost, overtime_pay."
                })

        output: Dict[str, Any] = {"results": results}
        if return_aggregate and grand_total > 0:
            output["aggregate_total"] = round(grand_total, 2)
            output["aggregate_currency"] = "PHP"
            output["aggregate_label"] = calculation_type

        return json.dumps(output, ensure_ascii=False, default=str)

    def _translate_to_english(self, text: str) -> str:
        """Translate query to English for better embedding match against English policy."""
        try:
            response = self.llm.invoke(
                "Translate the following HR question to English. "
                "If it is already in English, return it unchanged. "
                "Return only the translated question, nothing else.\n\n"
                f"Question: {text}"
            )
            translated = response.content.strip()
            if translated and translated != text:
                self.logger.info(f"Query translated: '{text}' → '{translated}'")
            return translated or text
        except Exception as e:
            self.logger.warning(f"Translation failed, using original query: {e}")
            return text

    def _tool_query_policy(self, question: str) -> str:
        if self.vectorstore is None:
            return json.dumps({"error": "Policy search is currently unavailable. ChromaDB not initialized."})

        try:
            search_query = self._translate_to_english(question)

            # k=6 for better coverage of distributed info (e.g. probation rules
            # spread across 22 leave-type sections)
            docs = self.vectorstore.max_marginal_relevance_search(
                search_query, k=6, fetch_k=25
            )

            if not docs:
                return json.dumps({
                    "policy_context": "No relevant policy information found.",
                    "citations": []
                })

            citations = []
            context_parts = []
            for i, doc in enumerate(docs):
                meta = doc.metadata
                section = meta.get("section_title", "HR Policy")
                excerpt = doc.page_content[:220] + "..." if len(doc.page_content) > 220 else doc.page_content
                citations.append({
                    "index": i + 1,
                    "section_title": section,
                    "policy_type": meta.get("policy_type", "general"),
                    "excerpt": excerpt
                })
                context_parts.append(
                    f"[Source {i + 1} — {section}]\n{doc.page_content}"
                )

            return json.dumps({
                "policy_context": "\n\n---\n\n".join(context_parts),
                "citations": citations,
                "instruction": (
                    "Answer the question strictly based on the policy_context above. "
                    "Reference sources like [Source 1] in your answer."
                )
            }, ensure_ascii=False)

        except Exception as e:
            self.logger.error(f"Policy query error: {e}")
            return json.dumps({"error": f"Policy search error: {str(e)}"})

    # ------------------------------------------------------------------
    # Agent construction
    # ------------------------------------------------------------------

    def _build_agent(self):
        tools = [
            StructuredTool.from_function(
                func=self._tool_get_employee_info,
                name="get_employee_info",
                description=(
                    "Retrieve employee data from the HR database. "
                    "Use for salary, leave balances (vacation, sick, service incentive), "
                    "position, rank, hire date, employment status, or organizational unit queries."
                ),
                args_schema=GetEmployeeInfoInput,
            ),
            StructuredTool.from_function(
                func=self._tool_calculate_hr_metrics,
                name="calculate_hr_metrics",
                description=(
                    "Perform HR-related calculations. Supports: "
                    "leave_encashment (payout for unused vacation + service incentive leave), "
                    "salary_raise (new salary after a percentage raise), "
                    "benefits_cost (cost of benefits at a given rate), "
                    "overtime_pay (overtime earnings with 25% premium). "
                    "Works for one or multiple employees. "
                    "Set return_aggregate=True when user asks for a combined total."
                ),
                args_schema=CalculateHRMetricsInput,
            ),
            StructuredTool.from_function(
                func=self._tool_query_policy,
                name="query_policy",
                description=(
                    "Search company HR policy documents. Use for ANY question about rules, "
                    "eligibility criteria, procedures, leave types, attendance policies, "
                    "probation rules, encashment rules, application processes, or policy violations."
                ),
                args_schema=QueryPolicyInput,
            ),
        ]

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_openai_tools_agent(self.llm, tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
            max_iterations=6,
        )

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------

    def _build_chat_history(self, state: ConversationState) -> list:
        history = []
        for msg in state.get_last_n(3):
            if msg["role"] == "user":
                history.append(HumanMessage(content=msg["content"]))
            else:
                history.append(AIMessage(content=msg["content"]))
        return history

    def _maybe_summarize(self, state: ConversationState):
        if not state.should_summarize() or len(state.messages) < 4:
            return
        try:
            to_summarize = state.messages[:-3]
            text = "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in to_summarize
            )
            prompt = (
                f"Previous summary: {state.summary or 'None'}\n\n"
                f"New conversation to incorporate into summary:\n{text}\n\n"
                "Write a concise updated summary. Keep key facts: employee names, "
                "queries asked, results provided, any important context."
            )
            response = self.llm.invoke(prompt)
            state.summary = response.content
            state.messages = state.messages[-3:]
            self.logger.info("Conversation summary updated.")
        except Exception as e:
            self.logger.error(f"Summarization error: {e}")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def get_response(self, query: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
        username = user_context.get("username", "unknown")

        # Inject user context so tools can read it
        self._current_user_context = user_context

        session = self._get_session(username)
        self._maybe_summarize(session)
        chat_history = self._build_chat_history(session)

        try:
            result = self.agent_executor.invoke({
                "input": query,
                "role": user_context.get("role", "employee"),
                "user_name": user_context.get("name", ""),
                "employee_id": user_context.get("username", ""),
                "last_mentioned_employees": (
                    ", ".join(session.last_mentioned_employees)
                    if session.last_mentioned_employees
                    else "None yet"
                ),
                "conversation_summary": state_summary(session),
                "chat_history": chat_history,
            })
        except Exception as e:
            self.logger.error(f"Agent execution error: {e}", exc_info=True)
            return {
                "answer": "An error occurred while processing your request. Please try again.",
                "steps": [],
                "citations": [],
            }

        answer = result.get("output", "")

        steps = []
        citations = []
        for action, observation in result.get("intermediate_steps", []):
            step = {
                "tool": action.tool,
                "input": action.tool_input,
                "output": observation,
            }
            steps.append(step)

            if action.tool == "query_policy":
                try:
                    obs_data = json.loads(observation)
                    citations.extend(obs_data.get("citations", []))
                except Exception:
                    pass

        session.add_message("user", query)
        session.add_message("assistant", answer)

        return {"answer": answer, "steps": steps, "citations": citations}

    def validate_user_access(self, user_context: Dict[str, Any], query: str) -> bool:
        """Lightweight auth check — real access control runs inside each tool."""
        return self.auth.is_authenticated


def state_summary(state: ConversationState) -> str:
    if state.summary:
        return state.summary
    if not state.messages:
        return "No previous conversation."
    recent = state.get_last_n(3)
    lines = [f"{m['role'].upper()}: {m['content'][:120]}" for m in recent]
    return "Recent context:\n" + "\n".join(lines)
