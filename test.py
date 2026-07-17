from causalab.tasks.natural_domains_arithmetic.config import NaturalDomainConfig
from causalab.tasks.natural_domains_arithmetic.causal_models import create_causal_model
from experiments.off_manifold import context_datasets as C
from collections import Counter

wd = create_causal_model(NaturalDomainConfig(domain_type='weekdays'))
mo = create_causal_model(NaturalDomainConfig(domain_type='months'))

context_prompts = C.create_contextual_prompts(mo, ['number'], n_per_context=3, seed=2, group_by = {"number": C.modclass_groups})
def print_trace(example):
    """Print a single counterfactual trace."""
    cf = example["counterfactual_inputs"][0]

    print(f"Base    : {b['raw_input']!r}")
    print(f"          entity={b['entity']}, number={b['number']}, result={b['result']}")
    print(f"CF      : {cf['raw_input']!r}")
    print(f"          entity={cf['entity']}, number={cf['number']}, result={cf['result']}")

for context, traces in context_prompts.items():
    print(f"Context: {context}")
    for trace in traces:
        for key, value in trace._values.items():
            print(f"  {key:<12}: {value}")