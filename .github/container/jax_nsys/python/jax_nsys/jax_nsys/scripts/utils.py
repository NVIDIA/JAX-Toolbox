def shuffle_analysis_arg(analysis):
    if analysis is None:
        return []
    # [Script(A), Arg(A1), Arg(A2), Script(B), Arg(B1)] becomes [[A, A1, A2], [B, B1]]
    out, current = [], []
    for t, x in analysis:
        if t == "script":
            if len(current):
                out.append(current)
            current = [x]
        else:
            assert t == "arg" and len(current)
            current.append(x)
    if len(current):
        out.append(current)
    return out