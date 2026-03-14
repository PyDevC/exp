import sqlglot
from sqlglot import exp


def convert_ast_to_dict(query: str) -> dict:
    tree = sqlglot.parse_one(query)

    result = {
        "columns": [],
        "Table": None,
        "operations": [],
        "op_pattern": []
    }

    # -------------------
    # Extract Columns
    # -------------------
    for expression in tree.expressions:
        if isinstance(expression, exp.Column):
            result["columns"].append(expression.name)

    # -------------------
    # Extract Table
    # -------------------
    if tree.args.get("from"):
        result["Table"] = tree.args["from"].this.name

    # -------------------
    # Extract WHERE operations
    # -------------------
    if tree.args.get("where"):
        where_expr = tree.args["where"].this
        _extract_operations(where_expr, result)

    return result


def _extract_operations(expr, result):
    """
    Recursively extract binary operations.
    """

    if isinstance(expr, exp.Binary):
        left = _get_operand(expr.left)
        right = _get_operand(expr.right)
        operator = expr.token_type.value

        result["operations"].append({
            "operator": operator,
            "left": left,
            "right": right
        })

        result["op_pattern"].append((left, right, operator))

    # Handle nested conditions like AND / OR
    if hasattr(expr, "left"):
        _extract_operations(expr.left, result)

    if hasattr(expr, "right"):
        _extract_operations(expr.right, result)


def _get_operand(node):
    if isinstance(node, exp.Column):
        return node.name
    if isinstance(node, exp.Literal):
        return node.this
    return node.sql()
