def format_number(value: float, decimals: int = 4) -> str:
    if abs(value) < 1e-10:
        return '0'
    if abs(value) >= 1000 or (abs(value) < 0.001 and value != 0):
        return f"{value:.{decimals-1}e}"
    return f"{value:.{decimals}f}"


def percent_error(actual: float, target: float) -> float:
    if abs(target) < 1e-10:
        return abs(actual) * 100
    return abs((actual - target) / target) * 100
