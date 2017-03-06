def boundary_exceedence_cost(s, cells):
    r = 0
    for cell in cells:
        if cell.temperature < cell.min_temp:
            r -=  (cell.min_temp - cell.temperature + 1) ** 4
        if cell.temperature > cell.max_temp:
            r -=  (cell.temperature - cell.max_temp  + 1) ** 4
    return r/100

def switch_cost(s, cells):
    r = 0
    for cell in cells:
        if not cell.time_on + cell.time_off - 1:
            r -= 1
    return r/100

def energy_price_cost(s, cells):
    realised = s['power_usage'] * s['pricing']
    return (1 - realised)/100
