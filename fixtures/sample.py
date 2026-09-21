import math

GLOBAL_RATE = 2


def total_price(items):
    subtotal = 0
    for item in items:
        if item.enabled:
            subtotal = subtotal + item.price()
    return math.ceil(subtotal * GLOBAL_RATE)
