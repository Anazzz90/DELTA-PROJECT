from core.fact_validator import FactValidator

validator = FactValidator()

facts = [
    'BTC volume surged 300%',
    'btc volume surged 300%',               # Duplicate (different case)
    'Trading volume plunged after 1 hour',  # Contradicts the first fact
    'It is unclear why this happened',      # Uncertain tier
    'Prices will likely continue rising'    # Inferred tier
]

result = validator.validate(facts)
print('\n=== VALIDATOR RESULTS ===')
print(f'Original count: {len(facts)}')
print(f'Duplicates removed: {result.duplicates_removed}')
print('\nTIERED FACTS:')
for f in result.facts:
    print(f' - [{f.tier.upper()}] {f.original}')

print('\nCONTRADICTIONS FOUND:')
for pair in result.contradictions:
    print(f'   {pair[0]}  <--->  {pair[1]}')
