import re
import ast
import operator
import streamlit as st

st.set_page_config(page_title="Multi-Agent Router Demo", page_icon="🧮", layout="wide")

# ----------------------------- Utility: Safe arithmetic evaluator -----------------------------
# Only allow arithmetic (+, -, *, /, **, %, parentheses). No variables, calls, attributes, etc.
ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
ALLOWED_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

class SafeEval(ast.NodeVisitor):
    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_type = type(node.op)
        if op_type in ALLOWED_BINOPS:
            return ALLOWED_BINOPS[op_type](left, right)
        raise ValueError("Unsupported binary operator")

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        op_type = type(node.op)
        if op_type in ALLOWED_UNARYOPS:
            return ALLOWED_UNARYOPS[op_type](operand)
        raise ValueError("Unsupported unary operator")

    def visit_Num(self, node):  # For Python <3.8
        return node.n

    def visit_Constant(self, node):  # For Python >=3.8
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed")

    def generic_visit(self, node):
        raise ValueError("Invalid expression: only numbers and + - * / ** % and parentheses are allowed")

def safe_eval_expr(expr: str) -> float:
    tree = ast.parse(expr, mode="eval")
    evaluator = SafeEval()
    return float(evaluator.visit(tree))

# ----------------------------- Router -----------------------------
MATH_KEYWORDS = [
    "add", "plus", "sum", "total",
    "minus", "subtract", "difference",
    "multiply", "times", "product",
    "divide", "over", "quotient",
    "power", "exponent", "mod", "remainder",
]

ARITH_CHARS = set("+-*/%^()")

def is_math_query(text: str) -> bool:
    t = text.lower().strip()
    if any(ch.isdigit() for ch in t):
        return True
    if any(k in t for k in MATH_KEYWORDS):
        return True
    if any(ch in ARITH_CHARS for ch in t):
        return True
    return False

# ----------------------------- Math Agent -----------------------------
def extract_numbers(text: str):
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", text)]

def math_agent(user_text: str, memory: dict):
    """
    Handles arithmetic calculations.
    Memory includes:
      - last_result: last numeric result
      - last_numbers: last list of numbers seen
    The agent supports continued calculations like "add 10", "multiply by 2" using last_result.
    """
    t = user_text.lower().strip()
    nums = extract_numbers(t)
    last_result = memory.get("last_result")

    def set_memory(result, nums_list):
        memory["last_result"] = result
        memory["last_numbers"] = nums_list

    # Patterns by intent keywords
    if any(k in t for k in ["add", "plus", "sum", "total"]):
        if nums:
            if len(nums) == 1 and last_result is not None:
                result = last_result + nums[0]
                set_memory(result, [last_result, nums[0]])
                return result, f"Added {nums[0]} to previous result {last_result}"
            else:
                result = sum(nums)
                set_memory(result, nums)
                return result, f"Summed numbers: {', '.join(map(str, nums))}"
        else:
            return None, "Please provide at least one number to add (or have a previous result)."

    if any(k in t for k in ["minus", "subtract", "difference"]):
        if nums:
            if len(nums) == 1 and last_result is not None:
                result = last_result - nums[0]
                set_memory(result, [last_result, nums[0]])
                return result, f"Subtracted {nums[0]} from previous result {last_result}"
            elif len(nums) >= 2:
                result = nums[0]
                for n in nums[1:]:
                    result -= n
                set_memory(result, nums)
                return result, f"Sequential subtraction: {', '.join(map(str, nums))}"
            else:
                return None, "Provide two numbers, or one number to subtract from the previous result."
        else:
            return None, "Please provide numbers to subtract."

    if any(k in t for k in ["multiply", "times", "product"]):
        if nums:
            if len(nums) == 1 and last_result is not None:
                result = last_result * nums[0]
                set_memory(result, [last_result, nums[0]])
                return result, f"Multiplied previous result {last_result} by {nums[0]}"
            else:
                result = 1.0
                for n in nums:
                    result *= n
                set_memory(result, nums)
                return result, f"Product of: {', '.join(map(str, nums))}"
        else:
            return None, "Please provide numbers to multiply."

    if any(k in t for k in ["divide", "over", "quotient"]):
        if nums:
            if len(nums) == 1 and last_result is not None:
                if nums[0] == 0:
                    return None, "Cannot divide by zero."
                result = last_result / nums[0]
                set_memory(result, [last_result, nums[0]])
                return result, f"Divided previous result {last_result} by {nums[0]}"
            elif len(nums) >= 2:
                result = nums[0]
                try:
                    for n in nums[1:]:
                        result /= n
                except ZeroDivisionError:
                    return None, "Cannot divide by zero."
                set_memory(result, nums)
                return result, f"Sequential division: {', '.join(map(str, nums))}"
            else:
                return None, "Provide two numbers, or one number to divide the previous result by."
        else:
            return None, "Please provide numbers to divide."

    if any(k in t for k in ["power", "exponent"]):
        if nums:
            if len(nums) == 1 and last_result is not None:
                result = last_result ** nums[0]
                set_memory(result, [last_result, nums[0]])
                return result, f"Raised previous result {last_result} to the power of {nums[0]}"
            elif len(nums) >= 2:
                base, exp = nums[0], nums[1]
                result = base ** exp
                set_memory(result, [base, exp])
                return result, f"Computed {base} ** {exp}"
            else:
                return None, "Provide exponent or base/exponent."
        else:
            return None, "Please provide a number for the exponent operation."

    if any(k in t for k in ["mod", "remainder"]):
        if nums:
            if len(nums) == 1 and last_result is not None:
                result = last_result % nums[0]
                set_memory(result, [last_result, nums[0]])
                return result, f"Computed previous result {last_result} mod {nums[0]}"
            elif len(nums) >= 2:
                result = nums[0] % nums[1]
                set_memory(result, nums[:2])
                return result, f"Computed {nums[0]} % {nums[1]}"
            else:
                return None, "Provide numbers for modulo operation."
        else:
            return None, "Please provide numbers for modulo operation."

    # If no keyword intent matched, try arithmetic expression (e.g., "45 + 67 * 2")
    # Allow referencing previous result via words 'ans', 'result', or 'prev'
    expr = t
    if last_result is not None:
        expr = re.sub(r"\b(ans|result|prev|previous)\b", str(last_result), expr)
    try:
        result = safe_eval_expr(expr)
        set_memory(result, nums if nums else [result])
        return result, f"Evaluated expression: {user_text}"
    except Exception:
        return None, "Could not parse the math expression. Try using numbers and + - * / ** % or keywords like add/multiply."

# ----------------------------- Knowledge Agent -----------------------------
KNOWLEDGE_BASE = {
    "langgraph": (
        "LangGraph is a Python library for building stateful, multi-actor workflows over LLMs using graph-based primitives. "
        "It lets you define Nodes (agents/tools), Edges (transitions), and manages state via a shared store, enabling features like "
        "routing, retries, timeouts, and human-in-the-loop. It's well-suited for multi-turn agents, tool orchestration, and complex flows."
    ),
    "streamlit": (
        "Streamlit is an open-source Python framework to build data apps quickly. It offers reactive UI components, session state, "
        "and simple primitives like st.write, st.chat_message, and st.sidebar for dashboards and interactive apps."
    ),
}

def knowledge_agent(user_text: str):
    t = user_text.lower().strip()
    # Try simple keyword lookup
    for key, val in KNOWLEDGE_BASE.items():
        if key in t:
            return val
    # Fallback generic answer
    return (
        "I'm a simple Knowledge Agent. Ask me about topics like 'LangGraph' or 'Streamlit'. "
        "For the demo, I return concise, built-in explanations."
    )

# ----------------------------- App State -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {role, content, agent, result, note}
if "math_memory" not in st.session_state:
    st.session_state.math_memory = {"last_result": None, "last_numbers": []}

# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    st.markdown("## ⚙️ Router & Memory")
    st.write("Router rule: If input includes numbers/math symbols or keywords, route to **Math Agent**; else to **Knowledge Agent**.")

    mm = st.session_state.math_memory
    st.markdown("### 🧠 Math Agent Memory")
    st.write(f"Last result: {mm.get('last_result')}")
    st.write(f"Last numbers: {mm.get('last_numbers')}")

    st.markdown("### 🧪 Try Examples")
    if st.button("Add 45 and 67"):
        st.session_state["preset"] = "Add 45 and 67"
    if st.button("Now tell me about LangGraph"):
        st.session_state["preset"] = "Now tell me about LangGraph"

# ----------------------------- Main Layout -----------------------------
st.title("🧮🔍 Multi-Agent Router Demo (Math + Knowledge)")

# Chat history display
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(f"**Agent:** {msg['agent']}")
        st.write(msg["content"])
        if msg.get("note"):
            st.caption(msg["note"])
        if msg.get("result") is not None:
            st.success(f"Result: {msg['result']}")

# Visible fallback: also show a simple Input / Output list so messages are
# readable even if chat widgets don't render in some environments.
st.markdown("---")
st.subheader("Input / Output (log)")
for msg in st.session_state.history:
    role = msg.get("role", "unknown").capitalize()
    agent = msg.get("agent", "")
    content = msg.get("content", "")
    note = msg.get("note")
    result = msg.get("result")
    line = f"**{role}** ({agent}): {content}"
    st.markdown(line)
    if note:
        st.caption(note)
    if result is not None:
        st.write(f"Result: {result}")

# Input
preset = st.session_state.pop("preset", None)
user_input = st.chat_input("Type a message… e.g., 'Add 45 and 67' or 'Tell me about LangGraph'", key="chat_input")
if preset and not user_input:
    user_input = preset

if user_input:
    # Log user message
    st.session_state.history.append({
        "role": "user",
        "content": user_input,
        "agent": "(routed)",
        "result": None,
        "note": None,
    })

    # Router decision
    route_to_math = is_math_query(user_input)
    routed_agent = "Math Agent" if route_to_math else "Knowledge Agent"

    # Process with selected agent
    if route_to_math:
        result, note = math_agent(user_input, st.session_state.math_memory)
        agent_text = (
            f"Using **Math Agent**. {note if note else ''}"
            if note else "Using **Math Agent**."
        )
        st.session_state.history.append({
            "role": "assistant",
            "content": agent_text,
            "agent": "Math Agent",
            "result": result,
            "note": "Router: math intent detected",
        })
    else:
        answer = knowledge_agent(user_input)
        st.session_state.history.append({
            "role": "assistant",
            "content": answer,
            "agent": "Knowledge Agent",
            "result": None,
            "note": "Router: general knowledge detected",
        })

    st.rerun()    python -m pip install --upgrade pip
    pip install streamlit
    streamlit run multi_agent_streamlit.py