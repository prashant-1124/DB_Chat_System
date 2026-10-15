import os
import json
import re
import calendar
from datetime import datetime

import psycopg
import ollama


# 1. CONFIGURATION


MODEL_NAME = "qwen2.5:3b"

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5432,
    "dbname": "Test_db",
    "user": "postgres",
    "password": "Postgres@11",
}

TABLE_NAME = "public.sample_workers"


# Only these columns can be used in SQL queries.
COLUMNS = {
    "emp_id": "emp_id",
    "name": "name",
    "number": "number",
    "occupation": "occupation",
    "addess": "addess",
    "dob": "dob",
    "date_of_registration": "date_of_registration",
    "salary": "salary",
    "gender": "gender",
    "social_category": "social_category",
    "city": "city",
}

GROUP_COLUMNS = list(COLUMNS.keys())
FILTER_COLUMNS = list(COLUMNS.keys())

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


# 2. DATABASE CONNECTION


def get_connection():
    if not DB_CONFIG["password"]:
        raise ValueError(
            "PostgreSQL password missing. "
            "Set the PGPASSWORD environment variable first."
        )

    return psycopg.connect(
        **DB_CONFIG,
        connect_timeout=10,
    )


# 3. INPUT CLEANING


def clean_arg(value):
    """Convert empty/null-like model arguments into None."""

    if value is None:
        return None

    if not isinstance(value, str):
        return value

    value = value.strip()

    if value.lower() in {
        "", "null", "none", "n/a", "not specified"
    }:
        return None

    return value


# 4. DATE FILTER HANDLING


def normalize_date_filter(value):
    """
    Supported examples:
      2026-03-15
      2026-03
      03-2026
      March 2026
      Mar 2026

    Returns:
      ("exact", "YYYY-MM-DD") for a specific date
      ("month", "YYYY-MM") for a month
    """

    value = clean_arg(value)

    if not value:
        return None

    value = str(value).strip()

    # Exact date
    try:
        dt = datetime.strptime(value, "%Y-%m-%d")
        return ("exact", dt.strftime("%Y-%m-%d"))
    except ValueError:
        pass

    # YYYY-MM
    match = re.fullmatch(r"(\d{4})-(\d{1,2})", value)

    if match:
        year = int(match.group(1))
        month = int(match.group(2))

        if 1 <= month <= 12:
            return ("month", f"{year:04d}-{month:02d}")

    # MM-YYYY
    match = re.fullmatch(r"(\d{1,2})-(\d{4})", value)

    if match:
        month = int(match.group(1))
        year = int(match.group(2))

        if 1 <= month <= 12:
            return ("month", f"{year:04d}-{month:02d}")

    # Month name + year
    for fmt in ("%B %Y", "%b %Y"):
        try:
            dt = datetime.strptime(value, fmt)
            return ("month", dt.strftime("%Y-%m"))
        except ValueError:
            continue

    return None


# 5. MAIN ANALYTICS FUNCTION


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
    Executes safe, read-only analytics against sample_workers.

    Examples:
      Salary of Kavita Rao where emp_id=12:
        metric=employee_details
        filter_column=name
        filter_value=Kavita Rao
        filter_column=emp_id
        filter_value=12

      Employee count by occupation:
        metric=count
        group_by=occupation

      Average salary by city:
        metric=average_salary
        group_by=city
    """

    metric = clean_arg(metric) or "employee_details"
    group_by = clean_arg(group_by)
    filter_column = clean_arg(filter_column)
    filter_value = clean_arg(filter_value)
    filter_mode = clean_arg(filter_mode) or "exact"
    date_column = clean_arg(date_column)
    date_value = clean_arg(date_value)

    metric = str(metric).lower()

    if metric not in METRICS:
        return {"error": f"Unsupported metric: {metric}"}

    if group_by:
        group_by = str(group_by).lower()

        if group_by not in GROUP_COLUMNS:
            return {"error": "Unsupported group-by column."}

    if filter_column:
        filter_column = str(filter_column).lower()

        if filter_column not in FILTER_COLUMNS:
            return {"error": "Unsupported filter column."}

    if filter_mode not in ("exact", "contains"):
        return {"error": "filter_mode must be exact or contains."}

    if date_column:
        date_column = str(date_column).lower()

        if date_column not in DATE_COLUMNS:
            return {
                "error": (
                    "Date filtering is supported only for "
                    "dob and date_of_registration."
                )
            }

    if date_value and not date_column:
        return {
            "error": "Provide date_column with date_value."
        }

    # Employee detail queries should return individual records,
    # rather than an aggregate.
    if metric == "employee_details" and group_by:
        return {
            "error": (
                "Use an aggregate metric when grouping employees."
            )
        }

    # Build SELECT expression from an allowlist.
    if metric == "employee_details":
        select_expr = ", ".join(
            f'"{column}"' for column in COLUMNS
        )

    elif metric == "count":
        select_expr = 'COUNT(DISTINCT "emp_id") AS "employee_count"'

    elif metric == "record_count":
        select_expr = 'COUNT(*) AS "record_count"'

    elif metric == "total_salary":
        select_expr = 'SUM("salary") AS "total_salary"'

    elif metric == "average_salary":
        select_expr = 'AVG("salary") AS "average_salary"'

    elif metric == "minimum_salary":
        select_expr = 'MIN("salary") AS "minimum_salary"'

    elif metric == "maximum_salary":
        select_expr = 'MAX("salary") AS "maximum_salary"'

    elif metric == "average_age":
        select_expr = (
            'AVG(EXTRACT(YEAR FROM AGE(CURRENT_DATE, "dob"))) '
            'AS "average_age"'
        )

    elif metric == "minimum_age":
        select_expr = (
            'MIN(EXTRACT(YEAR FROM AGE(CURRENT_DATE, "dob"))) '
            'AS "minimum_age"'
        )

    elif metric == "maximum_age":
        select_expr = (
            'MAX(EXTRACT(YEAR FROM AGE(CURRENT_DATE, "dob"))) '
            'AS "maximum_age"'
        )

    else:
        return {"error": "Unsupported metric."}

    if group_by and metric != "employee_details":
        sql = (
            f'SELECT "{group_by}", {select_expr} '
            f'FROM {TABLE_NAME}'
        )
    else:
        sql = f"SELECT {select_expr} FROM {TABLE_NAME}"
    conditions = []
    params = []

    # Ordinary filters: exact or partial text match.
    if filter_column and filter_value is not None:
        column_sql = f'"{filter_column}"'

        if filter_column == "emp_id":
            conditions.append(f"{column_sql}::text = %s")
            params.append(str(filter_value).strip())

        elif filter_mode == "contains":
            conditions.append(
                f"CAST({column_sql} AS TEXT) ILIKE %s"
            )
            params.append(f"%{filter_value}%")

        else:
            conditions.append(
                f"CAST({column_sql} AS TEXT) ILIKE %s"
        )
            params.append(str(filter_value))

    # Date filters: exact date or a whole month.
    if date_column and date_value:
        normalized = normalize_date_filter(date_value)

        if not normalized:
            return {
                "error": (
                    "Could not understand the date. Try "
                    "2026-03-15, March 2026, or 03-2026."
                )
            }

        date_type, normalized_value = normalized
        date_sql = f'"{date_column}"'

        if date_type == "exact":
            conditions.append(f"{date_sql}::date = %s::date")
            params.append(normalized_value)

        else:
            year, month = map(
                int, normalized_value.split("-")
            )

            conditions.append(
                f"{date_sql} >= %s::date "
                f"AND {date_sql} < %s::date"
            )

            params.append(f"{year:04d}-{month:02d}-01")

            if month == 12:
                next_month = f"{year + 1:04d}-01-01"
            else:
                next_month = (
                    f"{year:04d}-{month + 1:02d}-01"
                )

            params.append(next_month)

    if conditions:
        sql += " WHERE " + " AND ".join(conditions)

    # Add grouping only for aggregate metrics.
    if group_by:
        if metric == "employee_details":
            return {
                "error": (
                    "Grouping is not supported for employee_details."
                )
            }

        sql += f' GROUP BY "{group_by}"'
        sql += f' ORDER BY "{group_by}" NULLS LAST'

    elif metric == "employee_details":
        # Avoid returning an unlimited number of rows.
        sql += ' ORDER BY "name" NULLS LAST LIMIT 100'

    try:
        with get_connection() as conn:
            # Prevent accidental write queries in this transaction.
            with conn.transaction():
                conn.execute("SET TRANSACTION READ ONLY")

                with conn.cursor() as cur:
                    cur.execute(sql, params)

                    columns = [
                        description.name
                        for description in cur.description
                    ]

                    rows = cur.fetchall()

        results = [
            dict(zip(columns, row))
            for row in rows
        ]

        # Convert dates, decimals and other special values
        # into JSON-friendly values.
        results = json.loads(
            json.dumps(results, default=str)
        )

        return {
            "metric": metric,
            "group_by": group_by,
            "row_count": len(results),
            "results": results,
            "sql": sql,          # ← add this
            "params": params,    # ← add this
        }

    except Exception as exc:
        return {
            "error": str(exc),
            "sql": sql if 'sql' in locals() else None,
            "params": params if 'params' in locals() else None,
            "hint": (
                "Check the database connection, table name, "
                "column names and column data types."
            ),
        }


# 6. DESCRIBE THE DATABASE TABLE


def describe_data():
    sql = """
        SELECT
            column_name,
            data_type
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'sample_workers'
        ORDER BY ordinal_position;
    """

    try:
        with get_connection() as conn:
            with conn.transaction():
                conn.execute("SET TRANSACTION READ ONLY")

                with conn.cursor() as cur:
                    cur.execute(sql)
                    rows = cur.fetchall()

        return {
            "table": TABLE_NAME,
            "columns": [
                {
                    "column_name": row[0],
                    "data_type": row[1],
                }
                for row in rows
            ],
        }

    except Exception as exc:
        return {"error": str(exc)}


# 7. OLLAMA TOOL DEFINITIONS


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_workers",
                        "description": (
                            "Query the sample_workers PostgreSQL table. "
                            "Use employee_details to retrieve individual employee details. "
                            "Use count for distinct employees and record_count for rows. "
                            "Use total_salary, average_salary, minimum_salary, and "
                            "maximum_salary for salary calculations. "
                            "For maximum salary in Mumbai, use metric='maximum_salary', "
                            "filter_column='city', filter_value='Mumbai', and leave "
                            "group_by empty. Apply city filters whenever a city is specified. "
                            "Use group_by only when the user asks for results by each group. "
                            "For dates, use date_column and date_value. "
                            "Return the database results without inventing values."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "enum": METRICS,
                    },
                    "group_by": {
                        "type": "string",
                        "enum": [""] + GROUP_COLUMNS,
                    },
                    "filter_column": {
                        "type": "string",
                        "enum": [""] + FILTER_COLUMNS,
                    },
                    "filter_value": {
                        "type": "string",
                    },
                    "filter_mode": {
                        "type": "string",
                        "enum": ["exact", "contains"],
                    },
                    "date_column": {
                        "type": "string",
                        "enum": ["", "dob", "date_of_registration"],
                    },
                    "date_value": {
                        "type": "string",
                    },
                },
                "required": [
                    "metric",
                    "group_by",
                    "filter_column",
                    "filter_value",
                    "filter_mode",
                    "date_column",
                    "date_value",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_data",
            "description": (
                "Inspect the available columns and data types "
                "of the sample_workers table."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]



# 8. NATURAL LANGUAGE AGENT

SYSTEM_PROMPT = """
You are a PostgreSQL analytics assistant.

You may answer questions only using the sample_workers database
through the available tools.

Rules:
1. Never invent employee details, salaries, counts or results.
2. For an individual employee's salary or details, call
   analyze_workers with metric="employee_details".
3. For a question such as "What is the salary of Kavita Rao?",
   set filter_column="name", filter_value="Kavita Rao",
   filter_mode="exact", and leave group_by empty.
4. If a name may be spelled differently or the exact match
   returns no rows, you may retry using filter_mode="contains".e
5. For employee counts, use metric="count".
6. For average, total, minimum or maximum salary, use the
   corresponding salary metric.
7. For grouped summaries, set group_by to the requested column.
8. For date questions, use date_column and date_value.
9. If a tool returns an error, explain it honestly.
10. If there are multiple employees with the same name, show
    the matching records and explain that multiple matches exist.
11. If no matching record exists, say that no matching record
    was found. Never guess.
12. Do not execute or suggest arbitrary SQL through the tools.
13. Do not reveal personal contact numbers unless the user
    specifically asks for that field.
14. Format salary and other numeric results clearly and the currency we use is INR.
15. When the user asks for counts by social category,
    use group_by="social_category" and metric="count".
16. Preserve the exact social category names returned by
    the database. Never replace them with A, B, C or
    invent abbreviations.
17. Report every category and its employee count from
    the tool results.
18. If the user provides an emp_id, always filter using
    filter_column="emp_id" and the exact provided ID.
19. When a user asks for the salary of an employee with
    a specific emp_id, retrieve employee_details and return
    the salary for that exact record only.
20. Never sum or aggregate salary when the user asks for
    one employee's salary unless explicitly requested.
21. If multiple records match a name and no emp_id is given,
    show the matching employee IDs and ask which employee
    the user means.
22. If the user asks about the max city and didn't specify a metric like salary or count,dont guess. 
    Ask the user to clarify what they mean by "max city" and what metric they want to use.
23. Whenever the tool returns a successful result that contains a "sql" field,
    you MUST include the exact SQL query in your final answer.
    Present it clearly, for example:

    SQL executed:
    SELECT ...

24. If parameters were used, also show them (or the values that were substituted).
    Never invent or modify the SQL – copy it exactly from the tool result.
25. If the user asks for maximum salary in a city, call
    analyze_workers with metric="maximum_salary",
    filter_column="city", filter_value equal to the city name,
    filter_mode="exact", and leave group_by empty.

26. If the user asks for minimum salary in a city, use
    metric="minimum_salary" with the city filter.

27. If the user asks for average salary in a city, use
    metric="average_salary" with the city filter.

28. If the user asks which employee has the highest salary
    in a city, first identify the maximum salary and then
    retrieve employee_details for employees in that city
    whose salary matches that maximum.

29. Always show the actual SQL query returned by the tool
    and the query parameters. Never invent a query or result.
"""

def format_category_counts(result):
    if "error" in result:
        return f"Database error: {result['error']}"

    rows = result.get("results", [])
    sql = result.get("sql")
    params = result.get("params")

    if not rows:
        msg = "No matching employee records found."
    else:
        lines = ["Employee count by social category:"]
        for row in rows:
            category = row.get("social_category", "Unknown")
            count = row.get("employee_count", 0)
            lines.append(f"- {category}: {count} employees")
        msg = "\n".join(lines)

    # Always show the SQL when available
    if sql:
        msg += f"\n\nSQL executed:\n{sql}"
        if params:
            msg += f"\nParameters: {params}"

    return msg

def ask_agent(question):
    q = question.lower()

    category_question = (
        "social categor" in q
        and any(word in q for word in [
            "how many",
            "count",
            "each",
            "number",
            "category-wise",
            "category wise",
        ])
    )

    if category_question:
        result = analyze_workers(
            metric="count",
            group_by="social_category"
        )
        return format_category_counts(result)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for _ in range(6):
        response = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS,
        )

        assistant_message = response.message

        if not assistant_message.tool_calls:
            return (
                assistant_message.content
                or "No response received."
            )

        messages.append(assistant_message)

        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            arguments = tool_call.function.arguments or {}

            if function_name == "analyze_workers":
                result = analyze_workers(**arguments)

            elif function_name == "describe_data":
                result = describe_data()

            else:
                result = {"error": "Unknown tool requested."}

            messages.append({
                "role": "tool",
                "tool_name": function_name,
                "content": json.dumps(result, default=str),
            })

    return "Could not complete the request. Please try again."



# 9. START THE AGENT

if __name__ == "__main__":
    print("Ask questions about your employee data.")
    print("Type 'exit' to stop.\n")

    # Check database connection and table before starting.
    schema = describe_data()

    if "error" in schema:
        print("Database connection/table error:")
        print(schema["error"])
        raise SystemExit(1)

    print("Connected to PostgreSQL successfully.")
    print("Columns:", ", ".join(
        item["column_name"] for item in schema["columns"]
    ))

    while True:
        try:
            question = input("\nYou: ").strip()

            if question.lower() in ("exit", "quit", "stop"):
                print("Agent stopped.")
                break

            if not question:
                continue

            answer = ask_agent(question)
            print("\nAgent:", answer)

        except KeyboardInterrupt:
            print("\nAgent stopped.")
            break

        except Exception as exc:
            print(f"\nUnexpected error: {exc}")