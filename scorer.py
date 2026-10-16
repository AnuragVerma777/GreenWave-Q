"""Official Quantum Traffic Gridlock evaluator. Python 3.9+, standard library only.
Usage: python scorer.py --scenario scenario_R.json --schedule 10101010
       python scorer.py --scenario scenario_R.json --schedule schedule.json
Schedule order: A slots 1..4, then B slots 1..4; 1=NS, 0=EW.
This file evaluates schedules only; it does not construct a QUBO or optimize.
"""
import argparse
import json
from pathlib import Path


def decode_schedule(raw):
    if isinstance(raw, dict):
        if 'bits' in raw:
            return decode_schedule(raw['bits'])
        if set(raw) == {'A', 'B'}:
            return [*raw['A'], *raw['B']]
        raise ValueError('Schedule object must have bits or A and B arrays')
    if isinstance(raw, str):
        if len(raw) != 8 or any(c not in '01' for c in raw):
            raise ValueError('Schedule string must contain exactly eight 0/1 characters')
        return [int(c) for c in raw]
    if isinstance(raw, list):
        return raw
    raise ValueError('Invalid schedule format')


def validate_scenario(s):
    for key in ('arrivals', 'service_capacity', 'storage_capacity', 'overflow_weight', 'switch_weight', 'link'):
        if key not in s:
            raise ValueError('Missing scenario key: ' + key)
    for node in ('A', 'B'):
        for phase in ('NS', 'EW'):
            vals = s['arrivals'][node][phase]
            if len(vals) != 4 or any(type(x) is not int or x < 0 for x in vals):
                raise ValueError('Each arrival series must contain four nonnegative integers')
    if s['link']['from'] != 'A' or s['link']['to'] != 'B' or s['link']['phase'] not in ('NS','EW'):
        raise ValueError('Expected one A->B link with phase NS or EW')
    if not (0 <= s['link']['fraction'] <= 1):
        raise ValueError('Link fraction must be between 0 and 1')
    if type(s['service_capacity']) is not int or s['service_capacity'] < 1 or type(s['storage_capacity']) is not int or s['storage_capacity'] < 0:
        raise ValueError('Capacity and storage must be valid integers')
    if s['overflow_weight'] < 0 or s['switch_weight'] < 0:
        raise ValueError('Weights must be nonnegative')


def evaluate(scenario, schedule):
    validate_scenario(scenario)
    bits = decode_schedule(schedule)
    if len(bits) != 8 or any(type(b) is not int or b not in (0, 1) for b in bits):
        raise ValueError('Schedule must contain eight integer bits')
    plans = {'A': bits[:4], 'B': bits[4:]}
    queues = {n: {'NS': 0, 'EW': 0} for n in ('A', 'B')}
    pending = [0, 0, 0, 0]
    delay = overflow = 0
    for t in range(4):
        for node in ('A', 'B'):
            for phase in ('NS', 'EW'):
                inflow = scenario['arrivals'][node][phase][t]
                if node == 'B' and phase == scenario['link']['phase']:
                    inflow += pending[t]
                available = queues[node][phase] + inflow
                is_green = plans[node][t] == (1 if phase == 'NS' else 0)
                departures = min(available, scenario['service_capacity']) if is_green else 0
                queues[node][phase] = available - departures
                delay += queues[node][phase]
                overflow += max(0, queues[node][phase] - scenario['storage_capacity'])
                if node == 'A' and phase == scenario['link']['phase'] and t < 3:
                    # Exact decimal half-up for fraction with up to finite decimal representation.
                    from decimal import Decimal, ROUND_HALF_UP
                    amount = Decimal(str(scenario['link']['fraction'])) * Decimal(departures)
                    pending[t+1] += int(amount.quantize(Decimal('1'), rounding=ROUND_HALF_UP))
    switches = sum(plans[n][t] != plans[n][t-1] for n in ('A','B') for t in range(1,4))
    cost = delay + scenario['overflow_weight'] * overflow + scenario['switch_weight'] * switches
    return {'cost': round(cost, 6), 'delay': delay, 'overflow': overflow, 'switches': switches, 'final_queues': queues}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scenario', required=True)
    p.add_argument('--schedule', required=True, help='8 bits, or path to JSON schedule')
    a = p.parse_args()
    s = json.loads(Path(a.scenario).read_text())
    raw = json.loads(Path(a.schedule).read_text()) if Path(a.schedule).suffix.lower() == '.json' else a.schedule
    print(json.dumps(evaluate(s, raw), indent=2))

if __name__ == '__main__':
    main()
