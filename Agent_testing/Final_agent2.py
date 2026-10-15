import os
import json
import re
from datetime import datetime
from decimal import Decimal

import psycopg
import ollama


# =========================================================
# 1. CONFIGURATION
# =========================================================

MODEL_NAME = "qwen2.5:3b"

SCHEMA_NAME = "public"
TABLE_ONLY = "sample_workers"
TABLE_NAME = f"{SCHEMA_NAME}.{TABLE_ONLY}"

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5432,
    "dbname": "Test_db",
    "user": "postgres",
    # Uses the PGPASSWORD environment variable if set, else the default below.
    "password": os.getenv("PGPASSWORD", "Postgres@11"),
}

# Fallback column list. The real list is loaded from the database
# at startup (see load_columns), so a typo here will not break queries.
DEFAULT_COLUMNS = [
    "emp_id",
    "name",
    "number",
    "occupation",
    "addess",
    "dob",
    "date_of_registration",
    "salary",
    "gender",
    "social_category",
    "city",
]

COLUMNS = DEFAULT_COLUMNS.copy()
GROUP_COLUMNS = COLUMNS.copy()
FILTER_COLUMNS = COLUMNS.copy()
DATE_COLUMNS = {"dob", "date_of_registration"}

METRICS = [
    "employee_details",
    "count",
    "record_count",
    "total_salary",
    "average_salary",
    "minimum_salary",
    "maximum_salary",
    "average_age",
    "minimum_age",
    "maximum_age",
]

ALLOWED_ARGS = {
    "metric",
    "group_by",
    "filter_column",
    "filter_value",
    "filter_mode",
    "date_column",
    "date_value",
}


# =========================================================
# 2. DATABASE CONNECTION
# =========================================================

def get_connection():
    if not DB_CONFIG["password"]:
        raise ValueError(
            "PostgreSQL password is missing. Set PGPASSWORD "
            "or put the password in DB_CONFIG."
        )

    conn = psycopg.connect(**DB_CONFIG, connect_timeout=10)
    conn.read_only = True  # every query on this connection is read-only
    return conn


# =========================================================
# 3. INPUT CLEANING
# =========================================================

def clean_arg(value):
    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip()

        if value.lower() in {"", "null", "none", "n/a", "not specified"}:
            return None

    return value


# =========================================================
# 4. DATE FILTER HANDLING
# =========================================================

def normalize_date_filter(value):
    """
    Supported:
      2026-03-15
      2026-03
      03-2026
      March 2026
      Mar 2026
    """

    value = clean_arg(value)

    if not value:
        return None

    value = str(value).strip()

    try:
        dt = datetime.strptime(value, "%Y-%m-%d")
        return "exact", dt.strftime("%Y-%m-%d")
    except ValueError:
        pass

    match = re.fullmatch(r"(\d{4})-(\d{1,2})", value)
    if match:
        year, month = map(int, match.groups())
        if 1 <= month <= 12:
            return "month", f"{year:04d}-{month:02d}"

    match = re.fullmatch(r"(\d{1,2})-(\d{4})", value)
    if match:
        month, year = map(int, match.groups())
        if 1 <= month <= 12:
            return "month", f"{year:04d}-{month:02d}"

    for fmt in ("%B %Y", "%b %Y"):
        try:
            dt = datetime.strptime(value, fmt)
            return "month", dt.strftime("%Y-%m")
        except ValueError:
            continue

    return None


# =========================================================
# 5. SQL VERIFICATION DISPLAY
# =========================================================

def sql_literal(value):
    """Render a parameter as a SQL literal for display only."""

    if value is None:
        return "NULL"

    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"

    if isinstance(value, (int, float, Decimal)):
        return str(value)

    text = str(value)
    return "'" + text.replace("'", "''") + "'"


def render_sql_for_verification(sql, params):
    """
    Generate a copyable SQL statement for pgAdmin.
    This rendered SQL is never used for execution.
    """

    parts = sql.split("%s")

    if len(parts) - 1 != len(params):
        return None

    rendered = parts[0]

    for index, value in enumerate(params):
        rendered += sql_literal(value)
        rendered += parts[index + 1]

    return rendered


# =========================================================
# 6. READ-ONLY SQL EXECUTION
# =========================================================

def execute_read_only(sql, params=None):
    """Execute parameterized SQL and always return a dictionary."""

    if params is None:
        params = []

    verification_sql = render_sql_for_verification(sql, params)

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)

                if cur.description is None:
                    raise ValueError("The query did not return a result set.")

                column_names = [d.name for d in cur.description]
                rows = cur.fetchall()

        results = [dict(zip(column_names, row)) for row in rows]
        results = json.loads(json.dumps(results, default=str))

        return {
            "success": True,
            "results": results,
            "row_count": len(results),
            "sql": sql,
            "verification_sql": verification_sql,
            "params": params,
            "error": None,
        }

    except Exception as exc:
        return {
            "success": False,
            "results": [],
            "row_count": 0,
            "sql": sql,
            "verification_sql": verification_sql,
            "params": params,
            "error": f"{type(exc).__name__}: {exc}",
        }


# =========================================================
# 7. LOAD REAL COLUMNS FROM THE DATABASE
# =========================================================



def load_columns():
    """
    Read the real column names from the database so a wrong name
    in DEFAULT_COLUMNS (e.g. 'addess' vs 'address') cannot break queries.
    """

    global COLUMNS, GROUP_COLUMNS, FILTER_COLUMNS

    result = execute_read_only(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = %s
          AND table_name = %s
        ORDER BY ordinal_position;
        """,
        [SCHEMA_NAME, TABLE_ONLY],
    )

    if result.get("success") and result["results"]:
        COLUMNS = [row["column_name"] for row in result["results"]]
        GROUP_COLUMNS = COLUMNS.copy()
        FILTER_COLUMNS = COLUMNS.copy()

    return result




# =========================================================
# 8. FORMAT RESULTS AND SQL
# =========================================================

def append_sql_info(answer, result):
    verification_sql = result.get("verification_sql")
    sql = result.get("sql")
    params = result.get("params", [])

    if verification_sql:
        answer += "\n\nSQL Query (copy into pgAdmin):\n" + verification_sql

    if sql:
        answer += "\n\nParameterized SQL:\n" + sql
        answer += "\n\nParameters:\n" + json.dumps(params, default=str)

    return answer


def format_result(result):
    """Display database results and the corresponding SQL."""

    if not isinstance(result, dict):
        return (
            f"Unexpected result type: {type(result).__name__}. "
            "Expected a database result dictionary."
        )

    error = result.get("error")

    if error or result.get("success") is False:
        answer = f"Database error: {error or 'Unknown error'}"
    else:
        rows = result.get("results", [])

        if not rows:
            answer = "No matching records were found."
        elif len(rows) == 1:
            answer = json.dumps(rows[0], indent=2, ensure_ascii=False)
        else:
            answer = json.dumps(rows, indent=2, ensure_ascii=False)

    return append_sql_info(answer, result)


# =========================================================
# 9. ANALYTICS FUNCTION
# =========================================================

_COLUMNS_LOADED = False


def ensure_columns():
    global _COLUMNS_LOADED

    if not _COLUMNS_LOADED:
        result = load_columns()

        if result.get("success"):
            _COLUMNS_LOADED = True

def analyze_workers(
    metric="employee_details",
    group_by=None,
    filter_column=None,
    filter_value=None,
    filter_mode="exact",
    date_column=None,
    date_value=None,
):
    """
    Supports aggregate metrics, employee details, grouping,
    one standard filter, and an optional date/month filter.
    """

    metric = str(clean_arg(metric) or "employee_details").lower()
    group_by = clean_arg(group_by)
    filter_column = clean_arg(filter_column)
    filter_value = clean_arg(filter_value)
    filter_mode = str(clean_arg(filter_mode) or "exact").lower()
    date_column = clean_arg(date_column)
    date_value = clean_arg(date_value)

    if metric not in METRICS:
        return {"error": f"Unsupported metric: {metric}"}

    if group_by:
        group_by = str(group_by).lower()
        if group_by not in GROUP_COLUMNS:
            return {"error": f"Unsupported group-by column: {group_by}"}

    if filter_column:
        filter_column = str(filter_column).lower()
        if filter_column not in FILTER_COLUMNS:
            return {"error": f"Unsupported filter column: {filter_column}"}

    if filter_mode not in {"exact", "contains"}:
        return {"error": "filter_mode must be exact or contains."}

    if date_column:
        date_column = str(date_column).lower()
        if date_column not in DATE_COLUMNS or date_column not in COLUMNS:
            return {
                "error": (
                    "Date filtering is supported only for "
                    "dob and date_of_registration."
                )
            }

    if date_value and not date_column:
        return {"error": "Provide date_column with date_value."}

    if metric == "employee_details" and group_by:
        return {"error": "Grouping is not supported for employee_details."}

    salary = 'CAST("salary" AS NUMERIC)'
    age = 'EXTRACT(YEAR FROM AGE(CURRENT_DATE, CAST("dob" AS DATE)))'

    metric_sql = {
        "employee_details": ", ".join(f'"{c}"' for c in COLUMNS),
        "count": 'COUNT(DISTINCT "emp_id") AS "employee_count"',
        "record_count": 'COUNT(*) AS "record_count"',
        "total_salary": f'SUM({salary}) AS "total_salary"',
        "average_salary": f'ROUND(AVG({salary}), 2) AS "average_salary"',
        "minimum_salary": f'MIN({salary}) AS "minimum_salary"',
        "maximum_salary": f'MAX({salary}) AS "maximum_salary"',
        "average_age": f'ROUND(AVG({age}), 1) AS "average_age"',
        "minimum_age": f'MIN({age}) AS "minimum_age"',
        "maximum_age": f'MAX({age}) AS "maximum_age"',
    }

    select_expr = metric_sql[metric]

    if group_by:
        sql = f'SELECT "{group_by}", {select_expr} FROM {TABLE_NAME}'
    else:
        sql = f"SELECT {select_expr} FROM {TABLE_NAME}"

    conditions = []
    params = []

    # Standard filter.
    if filter_column and filter_value is not None:
        column_sql = f'"{filter_column}"'

        if filter_column == "emp_id":
            conditions.append(f"CAST({column_sql} AS TEXT) = %s")
            params.append(str(filter_value))

        elif filter_column == "gender" and str(filter_value).strip().lower() in {
            "m", "male", "man", "men", "f", "female", "woman", "women"
        }:
            # Matches both 'M'/'Male' and 'F'/'Female' style data.
            first_letter = str(filter_value).strip().lower()[0]
            conditions.append(f"CAST({column_sql} AS TEXT) ILIKE %s")
            params.append(f"{first_letter}%")

        elif filter_mode == "contains":
            conditions.append(f"CAST({column_sql} AS TEXT) ILIKE %s")
            params.append(f"%{filter_value}%")

        else:
            conditions.append(f"CAST({column_sql} AS TEXT) ILIKE %s")
            params.append(str(filter_value))

    # Date or month filter.
    if date_column and date_value:
        normalized = normalize_date_filter(date_value)

        if not normalized:
            return {
                "error": (
                    "Unrecognized date. Try 2026-03-15, "
                    "March 2026, or 03-2026."
                )
            }

        date_type, normalized_value = normalized
        date_sql = f'CAST("{date_column}" AS DATE)'

        if date_type == "exact":
            conditions.append(f"{date_sql} = %s::date")
            params.append(normalized_value)

        else:
            year, month = map(int, normalized_value.split("-"))

            if month == 12:
                next_month = f"{year + 1:04d}-01-01"
            else:
                next_month = f"{year:04d}-{month + 1:02d}-01"

            conditions.append(
                f"{date_sql} >= %s::date AND {date_sql} < %s::date"
            )
            params.extend([f"{year:04d}-{month:02d}-01", next_month])

    if conditions:
        sql += " WHERE " + " AND ".join(conditions)

    if group_by:
        sql += f' GROUP BY "{group_by}"'
        sql += f' ORDER BY "{group_by}" NULLS LAST'

    elif metric == "employee_details":
        order_col = "name" if "name" in COLUMNS else COLUMNS[0]
        sql += f' ORDER BY "{order_col}" NULLS LAST LIMIT 100'

    return execute_read_only(sql, params)


# =========================================================
# 10. DATABASE SCHEMA
# =========================================================

def describe_data():
    sql = """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = %s
          AND table_name = %s
        ORDER BY ordinal_position;
    """

    return execute_read_only(sql, [SCHEMA_NAME, TABLE_ONLY])


# =========================================================
# 11. SOCIAL CATEGORY COUNTS
# =========================================================

def format_category_counts(result):
    if not isinstance(result, dict):
        return f"Unexpected result type: {type(result).__name__}"

    if result.get("error") or not result.get("success"):
        return format_result(result)

    rows = result.get("results", [])

    if not rows:
        answer = "No matching employee records found."
    else:
        lines = ["Employee count by social category:"]

        for row in rows:
            category = row.get("social_category")
            count = row.get("employee_count", 0)
            lines.append(f"- {category}: {count} employees")

        answer = "\n".join(lines)

    return append_sql_info(answer, result)


# =========================================================
# 12. SPECIFIC OCCUPATION COUNT ROUTING
# =========================================================

def try_occupation_count(question):
    """
    Handle a specific occupation count using actual database values.
    Does not intercept requests for all occupation-wise counts.
    """

    if "occupation" not in COLUMNS:
        return None

    q = question.lower().strip()

    count_intent = any(
        phrase in q
        for phrase in [
            "count of",
            "how many",
            "number of",
            "count workers",
            "count employee",
            "count employees",
        ]
    )

    if not count_intent:
        return None

    group_intent = any(
        phrase in q
        for phrase in [
            "by occupation",
            "occupation-wise",
            "occupation wise",
            "each occupation",
            "every occupation",
            "all occupations",
        ]
    )

    if group_intent:
        return None

    occupation_result = execute_read_only(
        f"""
        SELECT DISTINCT "occupation"
        FROM {TABLE_NAME}
        WHERE "occupation" IS NOT NULL
        ORDER BY "occupation";
        """
    )

    if not occupation_result.get("success"):
        return format_result(occupation_result)

    occupations = [
        str(row["occupation"])
        for row in occupation_result["results"]
        if row.get("occupation") is not None
    ]

    matches = [
        occupation
        for occupation in occupations
        if occupation.strip()
        and re.search(
            r"(?<!\w)" + re.escape(occupation.lower()) + r"(?!\w)",
            q,
        )
    ]

    if not matches:
        return None

    longest_length = max(len(item) for item in matches)
    longest_matches = [i for i in matches if len(i) == longest_length]

    if len(longest_matches) != 1:
        return None

    result = analyze_workers(
        metric="count",
        filter_column="occupation",
        filter_value=longest_matches[0],
        filter_mode="exact",
    )

    return format_result(result)


# =========================================================
# 12B. GENERIC "COUNT BY <COLUMN>" ROUTING
# =========================================================

NON_GROUPABLE = {
    "emp_id", "name", "number", "address", "addess",
    "dob", "date_of_registration", "salary",
}


def detect_group_column(q):
    """Return the column the user wants a breakdown by, or None."""

    for col in COLUMNS:
        if col in NON_GROUPABLE:
            continue

        spaced = col.replace("_", " ")
        names = {spaced, spaced + "s"}

        if col == "city":
            names.add("cities")

        for name in names:
            n = re.escape(name)
            patterns = [
                rf"\b(?:by|per|each|every|across)\s+{n}\b",
                rf"\b{n}[\s-]*wise\b",
                rf"\bgroup(?:ed)?\s+by\s+{n}\b",
            ]

            if any(re.search(p, q) for p in patterns):
                return col

    return None


def try_group_count(question):
    q = question.lower().strip()

    count_intent = any(
        phrase in q for phrase in ["count", "how many", "number of"]
    )

    if not count_intent:
        return None

    # Salary/age questions are handled by the AI model.
    if any(word in q for word in ["salary", "age", "average", "maximum", "minimum"]):
        return None

    column = detect_group_column(q)

    if not column:
        return None

    result = analyze_workers(metric="count", group_by=column)

    if not result.get("success"):
        return format_result(result)

    rows = result.get("results", [])

    if not rows:
        answer = "No matching employee records found."
    else:
        label = column.replace("_", " ")
        lines = [f"Employee count by {label}:"]

        for row in rows:
            lines.append(
                f"- {row.get(column)}: {row.get('employee_count', 0)} employees"
            )

        answer = "\n".join(lines)

    return append_sql_info(answer, result)


# =========================================================
# 13. OLLAMA TOOL DEFINITIONS
# =========================================================

def build_tools():
    analytics_tool = {
        "type": "function",
        "function": {
            "name": "analyze_workers",
            "description": (
                "Query public.sample_workers. Use count for distinct "
                "employee IDs, record_count for rows, and the matching "
                "salary or age metric for aggregates. For a specific "
                "occupation count, set metric=count, filter_column="
                "occupation, filter_value to the occupation, and leave "
                "group_by empty. Use group_by=occupation only when the "
                "user asks for counts by every occupation. For salary "
                "in a city, apply a city filter and leave group_by empty "
                "unless the user explicitly asks for a city breakdown."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string", "enum": METRICS},
                    "group_by": {"type": "string", "enum": [""] + GROUP_COLUMNS},
                    "filter_column": {
                        "type": "string",
                        "enum": [""] + FILTER_COLUMNS,
                    },
                    "filter_value": {"type": "string"},
                    "filter_mode": {
                        "type": "string",
                        "enum": ["exact", "contains"],
                    },
                    "date_column": {
                        "type": "string",
                        "enum": [""] + sorted(DATE_COLUMNS),
                    },
                    "date_value": {"type": "string"},
                },
                "required": ["metric"],
            },
        },
    }

    schema_tool = {
        "type": "function",
        "function": {
            "name": "describe_data",
            "description": "Inspect sample_workers columns and types.",
            "parameters": {"type": "object", "properties": {}},
        },
    }

    return [analytics_tool, schema_tool]


# =========================================================
# 14. SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are a PostgreSQL worker-data analytics assistant.

Use database tools for data-dependent answers.

Rules:
1. Never invent database results.
2. Use employee_details for employee records.
3. Use count for distinct employee IDs.
4. Use record_count for table rows.
5. Use total_salary only when a sum is requested.
6. Use the matching average, minimum or maximum salary metric.
7. Use group_by only when a breakdown is requested.
8. For gender-wise counts, use count grouped by gender.
9. For social-category counts, use count grouped by social_category.
10. For occupation-wise counts, use count grouped by occupation.
11. For city-wise counts, use count grouped by city.
12. For a specific occupation count, filter occupation and leave
    group_by empty.
13. For maximum salary in a city, use maximum_salary and filter
    city. Leave group_by empty.
14. For minimum or average salary in a city, use the matching metric.
15. For a specific employee ID, filter by emp_id exactly.
16. If no records match, say so.
17. Use date_column and date_value for date/month questions.
18. Do not expose contact numbers unless requested.
19. Format salaries in INR when appropriate.
20. Never invent SQL or query results.
21. The application will show actual SQL and parameters.
22. "count by gender", "gender-wise count" or "worker count by gender"
    means group_by=gender with NO filter. Never turn a breakdown request
    into a filter such as gender=M.
23. Use a filter only when the user names a specific value, such as
    "male workers" or "workers in Delhi".
24. Only answer questions about the worker data. For anything else
    (programming, general knowledge, chat), do not answer and do not
    call any tool.
"""


# =========================================================
# 15. NATURAL LANGUAGE AGENT
# =========================================================

TOOLS = None  # built lazily after the real columns are loaded

def try_emp_id_lookup(question):
    q = question.lower()

    match = re.search(r"emp[_\s]*id\s*(?:is|=|:)?\s*(\d+)", q)

    if not match:
        return None

    result = analyze_workers(
        metric="employee_details",
        filter_column="emp_id",
        filter_value=match.group(1),
    )

    # Hide the contact number unless it was asked for.
    if result.get("success") and not any(
        w in q for w in ["number", "contact", "phone", "mobile"]
    ):
        for row in result["results"]:
            row.pop("number", None)

    return format_result(result)

OUT_OF_SCOPE_MESSAGE = (
    "I can only answer questions about the worker data in the database "
    "(for example: counts, salaries, ages, occupations, cities, or "
    "details of a specific employee). Please ask something related to "
    "the worker data."
)

SALARY_PATTERN = re.compile(
    r"\bsalar|\bpay|\bwage|\bearn|\bincome|\bctc\b|\bpaid\b"
)
AGE_PATTERN = re.compile(r"\bage\b|\bold|\byoung|\bborn")


def needs_clarification(question, args):
    """Stop the model from assuming salary or age when the user never said it."""

    q = question.lower()
    metric = str(clean_arg(args.get("metric")) or "").lower()

    if metric.endswith("_salary") and not SALARY_PATTERN.search(q):
        asked = True
    elif metric.endswith("_age") and not AGE_PATTERN.search(q):
        asked = True
    else:
        asked = False

    if not asked:
        return None

    return (
        "Your question doesn't say what to measure, so I won't guess.\n"
        "Please be more specific, for example:\n"
        "- maximum salary in Delhi\n"
        "- maximum age in Delhi\n"
        "- number of workers in Delhi\n"
        "- count of workers by city"
    )

METRIC_KEYWORDS = {
    "total": r"\btotal\b|\bsum\b|\boverall\b|\bcombined\b|\baltogether\b",
    "average": r"\baverage\b|\bavg\b|\bmean\b",
    "minimum": r"\bminimum\b|\bmin\b|\blowest\b|\bleast\b|\bsmallest\b|\byoungest\b",
    "maximum": r"\bmaximum\b|\bmax\b|\bhighest\b|\blargest\b|\bmost\b|\boldest\b",
}


def fix_metric(question, args):
    """
    Only keep an aggregate metric if the question really asks for it.
    Otherwise show the matching employee records.
    """

    q = question.lower()
    metric = str(clean_arg(args.get("metric")) or "").lower()

    kind = metric.split("_")[0]  # total / average / minimum / maximum

    if kind in METRIC_KEYWORDS and metric.endswith(("_salary", "_age")):
        if not re.search(METRIC_KEYWORDS[kind], q):
            has_filter = bool(
                clean_arg(args.get("filter_value"))
                or clean_arg(args.get("date_value"))
            )

            # A specific worker/date was named, so show those records.
            if has_filter:
                args["metric"] = "employee_details"
                args["group_by"] = None

    return args

def ask_agent(question):
    global TOOLS

    q = question.lower().strip()
    
    emp_answer = try_emp_id_lookup(question)

    if emp_answer is not None:
        return emp_answer

    # A. Specific occupation count.
    occupation_answer = try_occupation_count(question)

    if occupation_answer is not None:
        return occupation_answer

    # B. Social-category counts.
    category_question = (
        "social categor" in q
        and any(
            word in q
            for word in [
                "how many",
                "count",
                "each",
                "number",
                "category-wise",
                "category wise",
            ]
        )
    )

    if category_question:
        result = analyze_workers(metric="count", group_by="social_category")
        return format_category_counts(result)

    # B2. Any "count by / per / wise / each <column>" question.
    group_answer = try_group_count(question)

    if group_answer is not None:
        return group_answer

    # C. Natural-language tool calling.
    if TOOLS is None:
        TOOLS = build_tools()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for _ in range(6):
        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=messages,
                tools=TOOLS,
            )
        except Exception as exc:
            return (
                f"Ollama error: {type(exc).__name__}: {exc}\n"
                f"Check that Ollama is running and the model is installed:\n"
                f"  ollama pull {MODEL_NAME}"
            )

        assistant_message = response.message
        tool_calls = assistant_message.tool_calls or []

        if not tool_calls:
            return OUT_OF_SCOPE_MESSAGE

        messages.append(assistant_message)

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            arguments = tool_call.function.arguments or {}

            if function_name == "analyze_workers":
                if not isinstance(arguments, dict):
                    return "Invalid arguments returned by the AI model."

                # Drop any unexpected keys the small model may invent.
                safe_args = {
                    k: v for k, v in arguments.items() if k in ALLOWED_ARGS
                }
                
                safe_args = fix_metric(question, safe_args)

                clarification = needs_clarification(question, safe_args)

                if clarification:
                    return clarification
                
                clarification = needs_clarification(question, safe_args)

                if clarification:
                    return clarification

                try:
                    result = analyze_workers(**safe_args)
                except Exception as exc:
                    return f"Error running analysis: {type(exc).__name__}: {exc}"

                return format_result(result)

            if function_name == "describe_data":
                return format_result(describe_data())

            return f"The model requested an unknown tool: {function_name}"

    return "Could not complete the request. Please try again."


# =========================================================
# 16. COMMAND-LINE TEST
# =========================================================

if __name__ == "__main__":
    print("Worker Analytics Agent")
    print("Type 'exit' to stop.\n")

    schema = describe_data()

    if not schema.get("success"):
        print("Database connection/table error:")
        print(schema.get("error"))
        raise SystemExit(1)

    load_columns()
    TOOLS = build_tools()

    print("Connected to PostgreSQL successfully.")
    print("Columns:")

    for column in schema["results"]:
        print(f"- {column['column_name']} ({column['data_type']})")

    while True:
        try:
            question = input("\nYou: ").strip()

            if question.lower() in {"exit", "quit", "stop"}:
                print("Agent stopped.")
                break

            if not question:
                continue

            print("\nAgent:", ask_agent(question))

        except KeyboardInterrupt:
            print("\nAgent stopped.")
            break

        except Exception as exc:
            print(f"\nUnexpected error: {type(exc).__name__}: {exc}")
            
            
