# Mate Selection

A collection of mate selection methods for evolutionary algorithms

This packagge provides several strategies for selecting individuals from a
population to serve as parents in an evolutionary algorithm. Each individual
is represented by a reproductive fitness score, and a mate selection method
determines the probability with which individuals are selected.

* [**pypi.org**](https://pypi.org/project/mate_selection/)
* [**crates.io**](https://crates.io/crates/mate_selection)
* [**docs.rs**](https://docs.rs/mate_selection)

# Features

* Several mate-selection algorithms
* A unified API for all seven methods
* Sample individuals or mating pairs
* Probability Distribution (PDF) calculation
* Command-line argument parsing
* Python Support
* Rust Support

## Mate Selection Methods

Available methods range from uniform random selection to methods that strongly
favor individuals with high fitness. Both score-based and rank-based methods
are provided.

| Method              | Parameter            | Behavior                                     |
| :------------------ | :------------------- | :------------------------------------------- |
| `Random`            | —                    | Every individual has equal probability       |
| `Proportional`      | —                    | Probability proportional to score            |
| `Normalized`        | `cutoff`             | Standardizes scores before applying a cutoff |
| `Best`              | `count`              | Only the top `count` individuals reproduce   |
| `Percentile`        | `percentile`         | Excludes a fraction of the population        |
| `RankedLinear`      | `selection_pressure` | Linear weighting by rank                     |
| `RankedExponential` | `median`             | Exponential weighting by rank                |

## API Methods

| Method                   | Purpose                                    |
| :----------------------- | :----------------------------------------- |
| `pairs(amount, scores)`  | Randomly select mating pairs               |
| `select(amount, scores)` | Randomly select individual indices         |
| `pdf(scores)`            | Convert scores to normalized probabilities |
| `sample_weight(scores)`  | Convert scores to sampling weights         |

## Command-Line Argument Parsing

A **specification string** provides a simple text representation of a mate
selection method and its parameters. This is intended to make it easy to use
`mate_selection` in command-line applications, where selection methods can be
specified through command-line arguments or configuration files.

Selection methods can be converted to and from specification strings,
allowing a method to be serialized into a compact textual representation and
later reconstructed from that representation.

Selection methods also support serialization through serde (in rust) and
pickle (in python).

### Example Specification Strings:

```text
random
proportional
normalized=1.0
best=20
percentile=0.8
ranked-linear=0.5
ranked-exponential=10
```

## Random Numbers

The `mate_selection` package uses the [`rand`](https://github.com/rust-random/rand)
crate's thread-local random number generator (RNG). Currently this package
does not support alternate RNGs. Also, this package does not suport changing
the random seed, so all deterministic output is not currently possible.

All methods use
[stochastic_universal_sampling](https://github.com/D-McDougall/stochastic_universal_sampling)
to select individuals.

## NaN and Infinite Handling

tbd...


# Python

## Installation Instructions

Install the mate_selection package from PyPI using the command:

```bash
$ pip install mate_selection
```

## Example

```python
import mate_selection

selector = mate_selection.Proportional()

scores = [1.0, 2.0, 3.0, 4.0]

print(selector.pdf(scores))
print(selector.select(10, scores))
print(selector.pairs(5, scores))
```

# Rust

## Installation Instructions

Add the mate_selection package to your current Cargo.toml using the command:

```bash
$ cargo add mate_selection
```

## Example

```rust
use mate_selection::{MateSelection, Proportional};

let selector = Proportional();

let scores = vec![1.0, 2.0, 3.0, 4.0];

let probabilities = selector.pdf(scores.clone()).unwrap();
println!("{probabilities:?}");

let parents = selector.select(10, scores.clone()).unwrap();
println!("{parents:?}");

let pairs = selector.pairs(5, scores).unwrap();
println!("{pairs:?}");
```

# References

* Introduction to Evolutionary Computing  
  A.E. Eiben and J.E. Smith, 2003, 2015  
  <https://doi.org/10.1007/978-3-662-44874-8>  
  _(See chapter 5)_

# Copyright & License

Copyright 2024 David McDougall.

Licensed under the MIT No Attribution (MIT-0) license.
