def run(plugin, payload):
    handler = getattr(plugin, payload["name"])
    return handler(payload)


match payload:
    case {"kind": kind}:
        outcome = kind
