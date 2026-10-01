//! A collection of mate selection methods for evolutionary algorithms

use serde::{Deserialize, Serialize};
use std::fmt;

/// Mate selection algorithms randomly select pairs of individuals from a population.  
/// The sampling probability of each individuals is a function of its reproductive fitness or "score".  
pub trait MateSelection: std::fmt::Debug + Send + Sync {
    /// Choose multiple weighted pairs
    ///
    /// * Argument `amount` is the number of pairs to return.
    ///
    /// * Argument "scores" is a list containing the reproductive fitness of each individual.
    ///
    /// * Returns a list of pairs of parents to mate together.  
    ///   The parents are specified as indices into the scores list.
    ///
    /// This implementation almost never mates an individual with itself.
    fn pairs(&self, amount: usize, scores: Vec<f64>) -> Result<Vec<[usize; 2]>> {
        let mut pairs = self.select(amount * 2, scores)?;

        reduce_repeats(&mut pairs);

        Ok(transmute_vec_to_pairs(pairs))
    }

    /// Choose multiple weighted
    fn select(&self, amount: usize, scores: Vec<f64>) -> Result<Vec<usize>> {
        let rng = &mut rand::rng();

        if amount == 0 {
            return Ok(vec![]);
        } else {
            if scores.is_empty() {
                return Err(ArgumentError(format!("cannot select from empty set")));
            }
        }

        let weights = self.sample_weight(scores)?;

        Ok(stochastic_universal_sampling::choose_multiple_weighted(
            rng, amount, &weights,
        ))
    }

    /// Probability distribution function
    fn pdf(&self, scores: Vec<f64>) -> Result<Vec<f64>> {
        let mut pdf = self.sample_weight(scores)?;
        // Normalize the sum to one.
        let sum: f64 = pdf.iter().sum();
        let div_sum = 1.0 / sum;
        for x in pdf.iter_mut() {
            *x *= div_sum;
        }
        Ok(pdf)
    }

    /// Transform the reproductive fitness scores into sampling weights.  
    /// The sampling weights do **not** need to sum to one.
    fn sample_weight(&self, scores: Vec<f64>) -> Result<Vec<f64>>;
}

#[derive(Debug)]
pub struct ArgumentError(String);

type Result<T> = std::result::Result<T, ArgumentError>;

impl fmt::Display for ArgumentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ArgumentError {}

/// Select parents with a uniform random probability, ignoring the scores.
#[derive(Serialize, Deserialize, Debug, Copy, Clone, PartialEq)]
pub struct Random();

/// Select parents with a probability that is directly proportional to their score.
///
/// >   `probability(i) = score(i) / sum(score(x) for x in population)`
///
/// This method biases the selection based on the parents scores. This
/// method is significantly influenced by the magnitude of the fitness
/// scoring function, and by the signal-to-noise ratio between the average
/// score and the variations in the scores.
///
/// Negative or invalid (NaN) scores are discarded and those individuals are
/// not permitted to mate.
#[derive(Serialize, Deserialize, Debug, Copy, Clone, PartialEq)]
pub struct Proportional();

/// Normalize the fitness scores into a standard normal distribution.
/// First the scores are normalized into a standard distribution and then they
/// are shifted by the cutoff, which is naturally measured in standard deviations.
/// All scores which are less than the cutoff (now sub-zero) are
/// discarded and those individuals are not permitted to mate.
/// Finally the scores are divided by their sum to yield a selection probability.
/// This method improves upon the proportional method by controlling for the
/// magnitude and variation of the fitness scoring function.
///
/// Argument "**cutoff**" is the minimum negative deviation required for mating.
#[derive(Serialize, Deserialize, Debug, Copy, Clone, PartialEq)]
pub struct Normalized(pub f64);

/// Select parents from the best ranked individuals in the population.
/// Among the top scoring individuals, individuals are sampled with uniform
/// random probability.
///
/// Argument "**count**" is the number of individuals who are allowed to mate.
#[derive(Serialize, Deserialize, Debug, Copy, Clone, PartialEq)]
pub struct Best(pub usize);

/// Apply a simple percentile based threshold to the population.
/// Mating pairs are selected with uniform random probability from the eligible
/// members of the population.
///
/// Argument "**percentile**" is the fraction of the population which is denied
/// the chance to mate. At `0` everyone is allowed to mate and at `1` only the
/// single best individual is allowed to mate.
#[derive(Serialize, Deserialize, Debug, Copy, Clone, PartialEq)]
pub struct Percentile(pub f64);

/// Select parents based on their ranking in the population. This method sorts
/// the individuals by their scores in order to rank them from worst to best.
/// The sampling probability is a linear function of the rank.
/// >   `probability(rank) = (1/N) * (1 + SP - 2 * SP * (rank-1)/(N-1))`  
/// >   Where `N` is the population size, and  
/// >   Where `rank = 1` is the best individual and `rank = N` is the worst.  
///
/// Argument "**selection pressure**" measures the inequality in the probability
/// of being selected. Must be in the range [0, 1].
///
/// * At zero, all members are equally likely to be selected.  
/// * At one, the worst ranked individual will never be selected.  
#[derive(Serialize, Deserialize, Debug, Copy, Clone, PartialEq)]
pub struct RankedLinear(pub f64);

/// Select parents based on their ranking in the population, with an
/// exponentially weighted bias towards better ranked individuals. This method
/// can apply more selection pressure than the RankedLinear method can, which
/// is useful when dealing with very large populations or with a very large
/// number of offspring.
///
/// Argument "**median**" describes the exponential slope of the weights curve.
/// A small median will strongly favor the best individuals, whereas a
/// large median will sample the individuals more equally. The median is a
/// rank, and so it is naturally measured in units of individuals.
/// Approximately half of the sample will be drawn from individuals ranked
/// better than the median, and the other half will be selected from
/// individuals with a worse ranking than the median.
#[derive(Serialize, Deserialize, Debug, Copy, Clone, PartialEq)]
pub struct RankedExponential(pub usize);

/// Parse a `--selection` command line argument into a mate selection object.
///
/// The argument consists of the name of a mate selection method, optionally
/// followed by an equal sign `=` and a numeric parameter. Methods without
/// parameters are specified by name alone. Methods with parameters require
/// exactly one parameter of the appropriate type.
///
/// # Argument Syntax
/// 
/// ```text
/// random
/// proportional
/// normalized=<cutoff>
/// best=<count>
/// percentile=<percentile>
/// ranked-linear=<pressure>
/// ranked-exponential=<median>
/// ```
///
/// # Errors
///
/// Returns an error message if the argument can not be parsed, or if the
/// number is out of bounds.
pub fn parse(argument: &str) -> Box<dyn MateSelection> {
    match Argument::parse(argument).unwrap() {
        Argument::Random => Box::new(Random()),
        Argument::Proportional => Box::new(Proportional()),
        Argument::Normalized(cutoff) => Box::new(Normalized(cutoff)),
        Argument::Best(count) => Box::new(Best(count)),
        Argument::Percentile(percentile) => Box::new(Percentile(percentile)),
        Argument::RankedLinear(pressure) => Box::new(RankedLinear(pressure)),
        Argument::RankedExponential(median) => Box::new(RankedExponential(median)),
    }
}

#[derive(Debug, PartialEq)]
enum Argument {
    Random,
    Proportional,
    Best(usize),
    Percentile(f64),
    Normalized(f64),
    RankedLinear(f64),
    RankedExponential(usize),
}
impl Argument {
    fn parse(arg: &str) -> Result<Self> {
        let arg = arg.trim();

        if arg.is_empty() {
            return Err(ArgumentError("selection argument is empty".to_string()));
        }
        // Split the method-name and argument
        let (name, value) = match arg.split_once('=') {
            Some((name, value)) => (name.trim(), Some(value.trim())),
            None => (arg, None),
        };
        // Match the method-name and parse the argument
        match name {
            "random" => {
                if value.is_some() {
                    return Err(ArgumentError(
                        "`random` does not accept an argument".to_string(),
                    ));
                }
                Ok(Self::Random)
            }
            "proportional" => {
                if value.is_some() {
                    return Err(ArgumentError(
                        "`proportional` does not accept an argument".to_string(),
                    ));
                }
                Ok(Self::Proportional)
            }
            "normalized" => {
                let value = value
                    .ok_or_else(|| ArgumentError("`normalized` requires a cutoff".to_string()))?;
                let cutoff = value.parse::<f64>().map_err(|_| {
                    ArgumentError(format!("invalid cutoff for `normalized`: `{value}`"))
                })?;
                if !cutoff.is_finite() {
                    return Err(ArgumentError(
                        "`normalized` cutoff must be finite".to_string(),
                    ));
                }
                Ok(Self::Normalized(cutoff))
            }
            "best" => {
                let value =
                    value.ok_or_else(|| ArgumentError("`best` requires a count".to_string()))?;
                let count = value
                    .parse::<usize>()
                    .map_err(|_| ArgumentError(format!("invalid count for `best`: `{value}`")))?;
                if count == 0 {
                    return Err(ArgumentError(
                        "`best` requires a count greater than zero".to_string(),
                    ));
                }
                Ok(Self::Best(count))
            }
            "percentile" => {
                let value = value
                    .ok_or_else(|| ArgumentError("`percentile` requires a value".to_string()))?;
                let percentile = value
                    .parse::<f64>()
                    .map_err(|_| ArgumentError(format!("invalid percentile: `{value}`")))?;
                if !percentile.is_finite() || !(0.0..=1.0).contains(&percentile) {
                    return Err(ArgumentError(
                        "`percentile` must be a finite value in the range [0, 1]".to_string(),
                    ));
                }
                Ok(Self::Percentile(percentile))
            }
            "ranked-linear" => {
                let value = value.ok_or_else(|| {
                    ArgumentError("`ranked-linear` requires a selection pressure".to_string())
                })?;
                let pressure = value.parse::<f64>().map_err(|_| {
                    ArgumentError(format!(
                        "invalid selection pressure for `ranked-linear`: `{value}`"
                    ))
                })?;
                if !pressure.is_finite() || !(0.0..=1.0).contains(&pressure) {
                    return Err(ArgumentError(
                        "`ranked-linear` selection pressure must be a finite value in the range [0, 1]"
                            .to_string(),
                    ));
                }
                Ok(Self::RankedLinear(pressure))
            }
            "ranked-exponential" => {
                let value = value.ok_or_else(|| {
                    ArgumentError("`ranked-exponential` requires a median".to_string())
                })?;
                let median = value.parse::<usize>().map_err(|_| {
                    ArgumentError(format!(
                        "invalid median for `ranked-exponential`: `{value}`"
                    ))
                })?;
                if median == 0 {
                    return Err(ArgumentError(
                        "`ranked-exponential` median must be greater than zero".to_string(),
                    ));
                }
                Ok(Self::RankedExponential(median))
            }
            _ => Err(ArgumentError(format!("unknown selection method: `{name}`"))),
        }
    }
}

/// A collection of mate selection methods for evolutionary algorithms
///
/// Mate selection algorithms randomly select pairs of individuals from a
/// population. The sampling probability of each individuals is a function
/// of its reproductive fitness or "score".
///
/// These implementations almost never mate an individual with itself.
#[cfg(feature = "pyo3")]
#[pyo3::pymodule]
mod mate_selection {
    use super::MateSelection;
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    /// Select parents with a uniform random probability, ignoring the scores.
    #[pyclass]
    struct Random(super::Random);

    /// Select parents with a probability that is directly proportional to their score.
    ///
    /// >   probability(i) = score(i) / sum(score(x) for x in population)
    ///
    /// This method biases the selection based on the parents scores. This
    /// method is significantly influenced by the magnitude of the fitness
    /// scoring function, and by the signal-to-noise ratio between the average
    /// score and the variations in the scores.
    ///
    /// Negative or invalid (NaN) scores are discarded and those individuals are
    /// not permitted to mate.
    #[pyclass]
    struct Proportional(super::Proportional);

    /// Normalize the fitness scores into a standard normal distribution. First
    /// the scores are normalized into a standard distribution and then they
    /// are shifted by the cutoff, which is naturally measured in standard
    /// deviations. All scores which are less than the cutoff (now sub-zero)
    /// are discarded and those individuals are not permitted to mate. Finally
    /// the scores are divided by their sum to yield a selection probability.
    /// This method improves upon the proportional method by controlling for
    /// the magnitude and variation of the fitness scoring function.
    ///
    /// Argument "cutoff" is the minimum negative deviation required for mating.
    #[pyclass]
    struct Normalized(super::Normalized);

    /// Select parents from the best ranked individuals in the population.
    /// Among the top scoring individuals, individuals are sampled with uniform
    /// random probability.
    ///
    /// Argument "count" is the number of individuals who are allowed to mate.
    #[pyclass]
    struct Best(super::Best);

    /// Apply a simple percentile based threshold to the population.
    /// Mating pairs are selected with uniform random probability from the
    /// eligible members of the population.
    ///
    /// Argument "percentile" is the fraction of the population which is denied
    /// the chance to mate. At "0" everyone is allowed to mate and at "1" only the
    /// single best individual is allowed to mate.
    #[pyclass]
    struct Percentile(super::Percentile);

    /// Select parents based on their ranking in the population. This method
    /// sorts the individuals by their scores in order to rank them from worst
    /// to best. The sampling probability is a linear function of the rank.
    ///
    /// >   probability(rank) = (1/N) * (1 + SP - 2 * SP * (rank-1)/(N-1))  
    /// >   Where N is the population size, and  
    /// >   Where rank = 1 is the best individual and rank = N is the worst.  
    ///
    /// Argument "selection_pressure" measures the inequality in the
    /// probability of being selected. Must be in the range [0, 1].
    /// * At zero, all members are equally likely to be selected.
    /// * At one, the worst ranked individual will never be selected.
    #[pyclass]
    struct RankedLinear(super::RankedLinear);

    /// Select parents based on their ranking in the population, with an
    /// exponentially weighted bias towards better ranked individuals. This
    /// method can apply more selection pressure than the RankedLinear method
    /// can, which is useful when dealing with very large populations or with a
    /// very large number of offspring.
    ///
    /// Argument "median" describes the exponential slope of the weights curve.
    /// A small median will strongly favor the best individuals, whereas a
    /// large median will sample the individuals more equally. The median is a
    /// rank, and so it is naturally measured in units of individuals.
    /// Approximately half of the sample will be drawn from individuals ranked
    /// better than the median, and the other half will be selected from
    /// individuals with a worse ranking than the median.
    #[pyclass]
    struct RankedExponential(super::RankedExponential);

    #[pymethods]
    impl Random {
        #[new]
        fn new() -> Self {
            Self(super::Random())
        }
        fn __str__(&self) -> String {
            "mate_selection.Random()".to_string()
        }
        /// Choose multiple weighted pairs
        /// * Argument "amount" is the number of pairs to return.
        /// * Argument "scores" is the list of reproductive fitness scores.
        /// * Returns a list of pairs of parents to mate together.
        ///   The parents are specified as indices into the scores list.
        fn pairs(&self, amount: usize, scores: Vec<f64>) -> Vec<[usize; 2]> {
            self.0.pairs(amount, scores).unwrap()
        }
        /// Choose multiple weighted
        fn select(&self, amount: usize, scores: Vec<f64>) -> Vec<usize> {
            self.0.select(amount, scores).unwrap()
        }
        /// Probability distribution function
        fn pdf(&self, scores: Vec<f64>) -> Vec<f64> {
            <super::Random as MateSelection>::pdf(&self.0, scores).unwrap()
        }
    }

    #[pymethods]
    impl Proportional {
        #[new]
        fn new() -> Self {
            Self(super::Proportional())
        }
        fn __str__(&self) -> String {
            "mate_selection.Proportional()".to_string()
        }
        /// Choose multiple weighted pairs
        /// * Argument "amount" is the number of pairs to return.
        /// * Argument "scores" is the list of reproductive fitness scores.
        /// * Returns a list of pairs of parents to mate together.
        ///   The parents are specified as indices into the scores list.
        fn pairs(&self, amount: usize, scores: Vec<f64>) -> Vec<[usize; 2]> {
            self.0.pairs(amount, scores).unwrap()
        }
        /// Choose multiple weighted
        fn select(&self, amount: usize, scores: Vec<f64>) -> Vec<usize> {
            self.0.select(amount, scores).unwrap()
        }
        /// Probability distribution function
        fn pdf(&self, scores: Vec<f64>) -> Vec<f64> {
            <super::Proportional as MateSelection>::pdf(&self.0, scores).unwrap()
        }
    }

    #[pymethods]
    impl Normalized {
        #[new]
        fn new(cutoff: f64) -> PyResult<Self> {
            if cutoff.is_finite() {
                Ok(Self(super::Normalized(cutoff)))
            } else {
                Err(PyValueError::new_err(
                    "argument \"cutoff\" is not a finite number",
                ))
            }
        }
        fn __str__(&self) -> String {
            format!("mate_selection.Normalized({})", self.0 .0)
        }
        /// Choose multiple weighted pairs
        /// * Argument "amount" is the number of pairs to return.
        /// * Argument "scores" is the list of reproductive fitness scores.
        /// * Returns a list of pairs of parents to mate together.
        ///   The parents are specified as indices into the scores list.
        fn pairs(&self, amount: usize, scores: Vec<f64>) -> Vec<[usize; 2]> {
            self.0.pairs(amount, scores).unwrap()
        }
        /// Choose multiple weighted
        fn select(&self, amount: usize, scores: Vec<f64>) -> Vec<usize> {
            self.0.select(amount, scores).unwrap()
        }
        /// Probability distribution function
        fn pdf(&self, scores: Vec<f64>) -> Vec<f64> {
            <super::Normalized as MateSelection>::pdf(&self.0, scores).unwrap()
        }
    }

    #[pymethods]
    impl Best {
        #[new]
        fn new(count: usize) -> PyResult<Self> {
            if count > 0 {
                Ok(Self(super::Best(count)))
            } else {
                Err(PyValueError::new_err(
                    "argument \"count\" is less than one",
                ))
            }
        }
        fn __str__(&self) -> String {
            format!("mate_selection.Best({})", self.0 .0)
        }
        /// Choose multiple weighted pairs
        /// * Argument "amount" is the number of pairs to return.
        /// * Argument "scores" is the list of reproductive fitness scores.
        /// * Returns a list of pairs of parents to mate together.
        ///   The parents are specified as indices into the scores list.
        fn pairs(&self, amount: usize, scores: Vec<f64>) -> Vec<[usize; 2]> {
            self.0.pairs(amount, scores).unwrap()
        }
        /// Choose multiple weighted
        fn select(&self, amount: usize, scores: Vec<f64>) -> Vec<usize> {
            self.0.select(amount, scores).unwrap()
        }
        /// Probability distribution function
        fn pdf(&self, scores: Vec<f64>) -> Vec<f64> {
            <super::Best as MateSelection>::pdf(&self.0, scores).unwrap()
        }
    }

    #[pymethods]
    impl Percentile {
        #[new]
        fn new(percentile: f64) -> PyResult<Self> {
            if (0.0..=1.0).contains(&percentile) {
                Ok(Self(super::Percentile(percentile)))
            } else {
                Err(PyValueError::new_err(
                    "argument \"percentile\" is out of bounds [0, 1]",
                ))
            }
        }
        fn __str__(&self) -> String {
            format!("mate_selection.Percentile({})", self.0 .0)
        }
        /// Choose multiple weighted pairs
        /// * Argument "amount" is the number of pairs to return.
        /// * Argument "scores" is the list of reproductive fitness scores.
        /// * Returns a list of pairs of parents to mate together.
        ///   The parents are specified as indices into the scores list.
        fn pairs(&self, amount: usize, scores: Vec<f64>) -> Vec<[usize; 2]> {
            self.0.pairs(amount, scores).unwrap()
        }
        /// Choose multiple weighted
        fn select(&self, amount: usize, scores: Vec<f64>) -> Vec<usize> {
            self.0.select(amount, scores).unwrap()
        }
        /// Probability distribution function
        fn pdf(&self, scores: Vec<f64>) -> Vec<f64> {
            <super::Percentile as MateSelection>::pdf(&self.0, scores).unwrap()
        }
    }

    #[pymethods]
    impl RankedLinear {
        #[new]
        fn new(selection_pressure: f64) -> PyResult<Self> {
            if (0.0..=1.0).contains(&selection_pressure) {
                Ok(Self(super::RankedLinear(selection_pressure)))
            } else {
                Err(PyValueError::new_err(
                    "argument \"selection_pressure\" is out of bounds [0, 1]",
                ))
            }
        }
        fn __str__(&self) -> String {
            format!("mate_selection.RankedLinear({})", self.0 .0)
        }
        /// Choose multiple weighted pairs
        /// * Argument "amount" is the number of pairs to return.
        /// * Argument "scores" is the list of reproductive fitness scores.
        /// * Returns a list of pairs of parents to mate together.
        ///   The parents are specified as indices into the scores list.
        fn pairs(&self, amount: usize, scores: Vec<f64>) -> Vec<[usize; 2]> {
            self.0.pairs(amount, scores).unwrap()
        }
        /// Choose multiple weighted
        fn select(&self, amount: usize, scores: Vec<f64>) -> Vec<usize> {
            self.0.select(amount, scores).unwrap()
        }
        /// Probability distribution function
        fn pdf(&self, scores: Vec<f64>) -> Vec<f64> {
            <super::RankedLinear as MateSelection>::pdf(&self.0, scores).unwrap()
        }
    }

    #[pymethods]
    impl RankedExponential {
        #[new]
        fn new(median: usize) -> PyResult<Self> {
            if median > 0 {
                Ok(Self(super::RankedExponential(median)))
            } else {
                Err(PyValueError::new_err(
                    "argument \"median\" is less than one",
                ))
            }
        }
        fn __str__(&self) -> String {
            format!("mate_selection.RankedExponential({})", self.0 .0)
        }
        /// Choose multiple weighted pairs
        /// * Argument "amount" is the number of pairs to return.
        /// * Argument "scores" is the list of reproductive fitness scores.
        /// * Returns a list of pairs of parents to mate together.
        ///   The parents are specified as indices into the scores list.
        fn pairs(&self, amount: usize, scores: Vec<f64>) -> Vec<[usize; 2]> {
            self.0.pairs(amount, scores).unwrap()
        }
        /// Choose multiple weighted
        fn select(&self, amount: usize, scores: Vec<f64>) -> Vec<usize> {
            self.0.select(amount, scores).unwrap()
        }
        /// Probability distribution function
        fn pdf(&self, scores: Vec<f64>) -> Vec<f64> {
            <super::RankedExponential as MateSelection>::pdf(&self.0, scores).unwrap()
        }
    }

    #[pyfunction]
    fn parse<'py>(py: Python<'py>, argument: &str) -> PyResult<Bound<'py, PyAny>> {
        match super::Argument::parse(argument) {
            Ok(super::Argument::Random) => Ok(Bound::new(py, Random(super::Random()))?.into_any()),

            Ok(super::Argument::Proportional) => {
                Ok(Bound::new(py, Proportional(super::Proportional()))?.into_any())
            }

            Ok(super::Argument::Best(count)) => {
                Ok(Bound::new(py, Best(super::Best(count)))?.into_any())
            }

            Ok(super::Argument::Percentile(percentile)) => {
                Ok(Bound::new(py, Percentile(super::Percentile(percentile)))?.into_any())
            }

            Ok(super::Argument::Normalized(cutoff)) => {
                Ok(Bound::new(py, Normalized(super::Normalized(cutoff)))?.into_any())
            }

            Ok(super::Argument::RankedLinear(selection_pressure)) => Ok(Bound::new(
                py,
                RankedLinear(super::RankedLinear(selection_pressure)),
            )?
            .into_any()),

            Ok(super::Argument::RankedExponential(median)) => {
                Ok(Bound::new(py, RankedExponential(super::RankedExponential(median)))?.into_any())
            }

            Err(error) => Err(PyValueError::new_err(error.to_string())),
        }
    }
}

impl MateSelection for Random {
    fn sample_weight(&self, mut scores: Vec<f64>) -> Result<Vec<f64>> {
        scores.fill(1.0);
        Ok(scores)
    }

    fn pdf(&self, mut scores: Vec<f64>) -> Result<Vec<f64>> {
        if !scores.is_empty() {
            let p = 1.0 / scores.len() as f64;
            scores.fill(p);
        }
        Ok(scores)
    }

    fn select(&self, amount: usize, scores: Vec<f64>) -> Result<Vec<usize>> {
        let rng = &mut rand::rng();
        Ok(stochastic_universal_sampling::choose_multiple(
            rng,
            amount,
            scores.len(),
        ))
    }
}

impl MateSelection for Proportional {
    fn sample_weight(&self, mut scores: Vec<f64>) -> Result<Vec<f64>> {
        // Replace negative & invalid values with zero.
        for x in scores.iter_mut() {
            *x = x.max(0.0);
        }
        Ok(scores)
    }
}

impl MateSelection for Normalized {
    fn sample_weight(&self, mut scores: Vec<f64>) -> Result<Vec<f64>> {
        let cutoff = self.0;
        if !cutoff.is_finite() {
            return Err(ArgumentError(format!("argument \"cutoff\" is not finite")));
        }

        // Find and normalize by the average score.
        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        for x in scores.iter_mut() {
            *x -= mean;
        }
        // Find and normalize by the standard deviation of the scores.
        let var = scores.iter().map(|x| x.powi(2)).sum::<f64>() / scores.len() as f64;
        let std = var.sqrt();
        for x in scores.iter_mut() {
            // Shift the entire distribution and cutoff all scores which
            // are less than zero.
            *x = (*x / std - cutoff).max(0.0);
        }
        Ok(scores)
    }
}

fn arg_nth_max(amount: usize, data: &[f64]) -> Vec<usize> {
    if amount == 0 {
        return vec![];
    }
    let pivot = data.len() - amount;
    let mut data_copy = data.to_vec();
    let (_, cutoff, _) = data_copy.select_nth_unstable_by(pivot, f64::total_cmp);
    let cutoff = *cutoff;
    let mut index = Vec::with_capacity(amount);
    for (i, x) in data.iter().enumerate() {
        if *x >= cutoff {
            index.push(i)
        }
    }
    // Discard extra elements which are equal to the cutoff.
    if index.len() > amount {
        let mut num_discard = index.len() - amount;
        for cursor in (0..index.len()).rev() {
            if data[index[cursor]] == cutoff {
                index.swap_remove(cursor);
                num_discard -= 1;
                if num_discard == 0 {
                    break;
                }
            }
        }
    }
    index
}

fn zero_and_write_sparse(data: &mut [f64], index: &[usize], value: f64) {
    data.fill(0.0);
    for i in index {
        data[*i] = value;
    }
}

impl Best {
    fn args(&self) -> Result<usize> {
        let count = self.0;
        if count < 1 {
            return Err(ArgumentError(format!(
                "mate_selection.Best: argument `count` is less than one"
            )));
        }
        Ok(count)
    }
}
impl MateSelection for Best {
    fn select(&self, amount: usize, scores: Vec<f64>) -> Result<Vec<usize>> {
        if amount == 0 {
            return Ok(vec![]);
        } else {
            if scores.is_empty() {
                return Err(ArgumentError(format!("cannot select from empty set")));
            }
        }
        let rng = &mut rand::rng();
        let num_best = self.args()?;
        let index = arg_nth_max(num_best, &scores);
        let sample = stochastic_universal_sampling::choose_multiple(rng, amount, index.len());
        Ok(sample.iter().map(|&s| index[s]).collect())
    }
    fn pdf(&self, mut scores: Vec<f64>) -> Result<Vec<f64>> {
        let num_best = self.args()?;
        let index = arg_nth_max(num_best, &scores);
        zero_and_write_sparse(&mut scores, &index, 1.0 / num_best as f64);
        Ok(scores)
    }
    fn sample_weight(&self, mut scores: Vec<f64>) -> Result<Vec<f64>> {
        let num_best = self.args()?;
        let index = arg_nth_max(num_best, &scores);
        zero_and_write_sparse(&mut scores, &index, 1.0);
        Ok(scores)
    }
}

impl Percentile {
    fn get_index(&self, scores: &[f64]) -> Result<Vec<usize>> {
        let percentile = self.0;
        if !(0.0..=1.0).contains(&percentile) {
            return Err(ArgumentError(format!(
                "argument \"percentile\" is out of bounds [0, 1]"
            )));
        }
        let num_eligible = ((1.0 - percentile) * scores.len() as f64).round() as usize;
        Ok(arg_nth_max(num_eligible.max(1), scores))
    }
}
impl MateSelection for Percentile {
    fn select(&self, amount: usize, scores: Vec<f64>) -> Result<Vec<usize>> {
        let rng = &mut rand::rng();
        let index = self.get_index(&scores)?;
        let sample = stochastic_universal_sampling::choose_multiple(rng, amount, index.len());
        Ok(sample.iter().map(|&s| index[s]).collect())
    }
    fn pdf(&self, mut scores: Vec<f64>) -> Result<Vec<f64>> {
        let index = self.get_index(&scores)?;
        zero_and_write_sparse(&mut scores, &index, 1.0 / index.len() as f64);
        Ok(scores)
    }
    fn sample_weight(&self, mut scores: Vec<f64>) -> Result<Vec<f64>> {
        let index = self.get_index(&scores)?;
        zero_and_write_sparse(&mut scores, &index, 1.0);
        Ok(scores)
    }
}

impl MateSelection for RankedLinear {
    fn sample_weight(&self, mut scores: Vec<f64>) -> Result<Vec<f64>> {
        let selection_pressure = self.0;
        if !(0.0..=1.0).contains(&selection_pressure) {
            return Err(ArgumentError(format!(
                "argument \"selection_pressure\" is out of bounds [0, 1]"
            )));
        }

        let div_n = if scores.len() == 1 {
            0.0 // Value does not matter, just don't crash.
        } else {
            1.0 / (scores.len() - 1) as f64
        };
        for (rank, index) in argsort(&scores).iter().enumerate() {
            // Reverse the ranking from ascending to descending order
            // so that rank 0 is the best & rank N-1 is the worst.
            let rank = scores.len() - 1 - rank;
            // Scale the ranking into the range [0, 1].
            let rank = rank as f64 * div_n;
            scores[*index] = 1.0 + selection_pressure - 2.0 * selection_pressure * rank;
        }
        Ok(scores)
    }
}

impl MateSelection for RankedExponential {
    fn sample_weight(&self, mut scores: Vec<f64>) -> Result<Vec<f64>> {
        let median = self.0;
        if median < 1 {
            return Err(ArgumentError(format!(
                "argument \"median\" is less than one"
            )));
        }
        for (rank, index) in argsort(&scores).iter().enumerate() {
            let rank = scores.len() - rank - 1;
            scores[*index] = (-(2.0_f64.ln()) * rank as f64 / median as f64).exp();
        }
        Ok(scores)
    }
}

fn argsort(scores: &[f64]) -> Vec<usize> {
    let mut argsort: Vec<_> = (0..scores.len()).collect();
    argsort.sort_unstable_by(|a, b| f64::total_cmp(&scores[*a], &scores[*b]));
    argsort
}

/// This helps avoid mating an individual with itself.
fn reduce_repeats(data: &mut [usize]) {
    debug_assert!(is_even(data.len()));
    // Simple quadratic greedy algorithm for breaking up pairs of repeated elements.
    // First search for pairs of repeated values.
    'outer: for cursor in (0..data.len()).step_by(2) {
        let value = data[cursor];
        if value == data[cursor + 1] {
            // Then find a different value to swap with.
            for search in (cursor + 2..data.len()).step_by(2) {
                if data[search] != value && data[search + 1] != value {
                    data.swap(cursor, search);
                    continue 'outer;
                }
            }
            for search in (0..cursor).step_by(2) {
                if data[search] != value && data[search + 1] != value {
                    data.swap(cursor, search);
                    continue 'outer;
                }
            }
        }
    }
}

/// Transmute the vector of samples into pairs of samples, without needlessly copying the data.
fn transmute_vec_to_pairs(data: Vec<usize>) -> Vec<[usize; 2]> {
    // Check that there are an even number of values in the vector.
    assert!(is_even(data.len()));
    // Check the data alignment.
    assert_eq!(
        std::mem::align_of::<usize>(),
        std::mem::align_of::<[usize; 2]>()
    );
    // Take manual control over the data vector.
    let mut data = std::mem::ManuallyDrop::new(data);
    unsafe {
        // Disassemble the vector.
        let ptr = data.as_mut_ptr();
        let mut len = data.len();
        let mut cap = data.capacity();
        // Transmute the vector.
        let ptr = std::mem::transmute::<*mut usize, *mut [usize; 2]>(ptr);
        len /= 2;
        cap /= 2;
        // Reassemble and return the data.
        Vec::from_raw_parts(ptr, len, cap)
    }
}

const fn is_even(x: usize) -> bool {
    x & 1 == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flatten_and_sort(pairs: &Vec<[usize; 2]>) -> Vec<usize> {
        let mut data: Vec<usize> = pairs.iter().flatten().copied().collect();
        data.sort_unstable();
        data
    }

    #[test]
    fn is_even() {
        assert!(super::is_even(0));
        assert!(!super::is_even(1));
        assert!(super::is_even(2));
        assert!(!super::is_even(3));
    }

    #[test]
    fn no_data() {
        let pairs = Proportional().pairs(0, vec![]).unwrap();
        assert!(pairs.is_empty());

        let pairs = Proportional().pairs(0, vec![1.0, 2.0, 3.0]).unwrap();
        assert!(pairs.is_empty());
    }

    #[test]
    fn truncate_top_one() {
        // Truncate all but the single best individual.
        let algo = Percentile(0.99);
        let weights: Vec<f64> = (0..100).map(|x| x as f64 / 100.0).collect();
        let pairs = algo.pairs(1, weights).unwrap();
        assert!(pairs == [[99, 99]]);
    }

    #[test]
    fn truncate_top_two() {
        // Truncate all but the best two individuals.
        let algo = Percentile(0.98);
        let weights: Vec<f64> = (0..100).map(|x| x as f64 / 100.0).collect();
        let pairs = algo.pairs(1, weights).unwrap();
        assert!(pairs == [[98, 99]] || pairs == [[99, 98]]);
    }

    #[test]
    fn truncate_none() {
        // Truncate none of the individuals.
        let algo = Percentile(0.0);
        let weights: Vec<f64> = (0..100).map(|x| x as f64 / 100.0).collect();
        let pairs = algo.pairs(50, weights).unwrap();
        let selected = flatten_and_sort(&pairs);
        assert_eq!(selected, (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn truncate_all() {
        // Truncating all individuals should actually just return the single
        // best individual. This situation happens when building the starting
        // population.
        let algo = Percentile(0.999_999_999); // Technically less than one.
        let weights: Vec<f64> = (0..100).map(|x| x as f64 / 100.0).collect();
        let pairs = algo.pairs(1, weights).unwrap();
        assert!(pairs == [[99, 99]]);
    }

    #[test]
    fn all_equal_to_the_best() {
        Best(3).select(1, vec![4.0, 4.0, 4.0, 4.0]).unwrap();
    }

    #[test]
    fn propotional() {
        // All scores are equal, proportional should select all of the items.
        let weights = vec![1.0; 10];
        let algo = Proportional();
        let selected = flatten_and_sort(&algo.pairs(5, weights).unwrap());
        assert_eq!(selected, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn propotional_outlier() {
        // Index 0 is an outlier. Proportional selection should allow the
        // outlier to dominate the sample. The other items should not be selected.
        let weights = vec![1000_000_000_000_000.0, 1.0, 1.0, 1.0];
        let algo = Proportional();
        let selected = flatten_and_sort(&algo.pairs(10, weights).unwrap());
        let inliers: Vec<_> = selected.iter().filter(|&idx| *idx != 0).collect();
        assert!(inliers.is_empty());
    }

    #[test]
    fn propotional_negative() {
        // One score is extremely negative and another is NAN.
        // Proportional should ignore them.
        let mut weights = vec![1.0; 12];
        weights[5] = -100.0;
        weights[6] = f64::NAN;
        let algo = Proportional();
        let selected = flatten_and_sort(&algo.pairs(5, weights).unwrap());
        assert_eq!(selected, [0, 1, 2, 3, 4, 7, 8, 9, 10, 11]);
    }

    #[test]
    fn normalized() {
        // Normalize can deal with negative scores, it does not care about their absolute values.
        let weights = vec![-20.0, -12.0, -11.0, -10.5, -10.0, -9.5, -9.0, -8.0, 0.0];
        const MEAN_IDX: usize = 4;
        const MAX_IDX: usize = 8;
        let cutoff = -0.01;
        let algo = Normalized(cutoff);
        let selected = flatten_and_sort(&algo.pairs(2, weights).unwrap());
        // Only scores greater than the mean should have been selected.
        assert!(selected.iter().all(|&x| x >= MEAN_IDX));
        // The sample should contain the highest score, but not be dominated by it.
        assert!(selected.contains(&MAX_IDX));
        assert!(!selected.iter().all(|&x| x == MAX_IDX));
    }

    #[test]
    fn ranked_linear() {
        // Index 0 is an outlier.
        // Ranking the scores should prevent the outlier from dominating.
        let weights = vec![1000_000_000_000_000.0, 1.0, 1.0, 1.0];

        // No selection pressure, should select all four scores.
        let algo = RankedLinear(0.0);
        let selected = flatten_and_sort(&algo.pairs(2, weights).unwrap());
        assert_eq!(selected, vec![0, 1, 2, 3]);
    }

    /// Finds those off-by-one errors.
    #[test]
    fn ranked_linear_single() {
        let weights = vec![4.0];
        let algo = RankedLinear(0.5);
        let selected = flatten_and_sort(&algo.pairs(1, weights).unwrap());
        assert_eq!(selected, vec![0, 0]);
    }

    #[test]
    fn ranked_linear_outlier() {
        // Index 0 is an outlier.
        // Ranking the scores should prevent the outlier from dominating.
        let mut weights = vec![1000_000_000_000_000.0];
        weights.append(&mut vec![1.0; 1000]);
        // With selection pressure, the outlier still should not dominate the sampling.
        let algo = RankedLinear(1.0);
        let selected = flatten_and_sort(&algo.pairs(10, weights).unwrap());
        let inliers: Vec<_> = selected.iter().filter(|&idx| *idx != 0).collect();
        assert!(!inliers.is_empty());
    }

    #[test]
    fn ranked_exponential() {
        let test_cases = [
            (1, 1, 2, 99), // Test selecting with one single weight does not crash.
            (3, 1, 4, 1),
            (100, 10, 100, 5),
            (1000, 10, 100, 5),
            (10_000, 100, 10_000, 20),
            (10_000, 1000, 10_000, 50),
        ];
        for (num, median, sample, tolerance) in test_cases {
            let weights: Vec<f64> = (0..num).map(|x| x as f64).collect();
            let algo = RankedExponential(median);
            assert_eq!(sample, (sample / 2) * 2); // Sample count needs to be even for this to work.
            let selected = flatten_and_sort(&algo.pairs(sample / 2, weights).unwrap());
            dbg!(&selected);
            // Count how many elements are from the top ranked individuals.
            let top_count_actual = selected
                .iter()
                .filter(|&&idx| idx >= (num - median))
                .count();
            let top_count_desired = sample / 2;
            dbg!(num, median, sample, tolerance);
            dbg!(top_count_actual, top_count_desired);
            assert!((top_count_actual as i64 - top_count_desired as i64).abs() <= tolerance);
            assert!(top_count_actual > 0);
        }
    }

    /// Check that this avoids mating individuals with themselves.
    #[test]
    fn pairs() {
        // N is the population size.
        // P is the number of mating pairs.
        // R is the percent of the pairs that are duplicates.
        for (n, max_r) in [
            (2, 5.0),
            (3, 4.0),
            (4, 3.0),
            (5, 3.0),
            (10, 3.0),
            (20, 2.0),
            (100, 1.0),
            //
        ] {
            let p = 10 * n;
            // let p = 3;
            let indices = Random().pairs(p, vec![1.0; n]).unwrap();
            let num_repeats = indices.iter().filter(|[a, b]| a == b).count();
            let percent_repeats = 100.0 * num_repeats as f64 / indices.len() as f64;

            println!("Population Size = {n}, Mating Pairs = {p}, Repeats = {percent_repeats:.2} %");
            dbg!(indices);
            assert!(percent_repeats <= max_r);
        }
    }

    /// Example of the trait used as an argument.
    #[test]
    fn argument() {
        fn foobar(select: &dyn MateSelection) {
            dbg!(select.select(0, vec![])).unwrap();
        }
        let x: &dyn MateSelection = if rand::random() {
            &Random()
        } else {
            &Proportional()
        };
        foobar(x);
    }

    /// Example of sending a trait object to a new thread.
    #[test]
    fn threading() {
        let trait_object: Box<dyn MateSelection> = Box::new(Random());
        let handle = std::thread::spawn(move || {
            dbg!(trait_object.select(0, vec![])).unwrap();
        });
        handle.join().unwrap();
    }

    #[test]
    fn parse_valid_arguments() {
        let test_cases = [
            ("random", Argument::Random),
            ("proportional", Argument::Proportional),
            ("normalized=1.5", Argument::Normalized(1.5)),
            ("best=20", Argument::Best(20)),
            ("percentile=0.25", Argument::Percentile(0.25)),
            ("ranked-linear=0.75", Argument::RankedLinear(0.75)),
            ("ranked-exponential=10", Argument::RankedExponential(10)),
            ("  random  ", Argument::Random),
            ("  best = 20  ", Argument::Best(20)),
            ("percentile=0", Argument::Percentile(0.0)),
            ("percentile=1", Argument::Percentile(1.0)),
            ("ranked-linear=0", Argument::RankedLinear(0.0)),
            ("ranked-linear=1", Argument::RankedLinear(1.0)),
        ];

        for (input, expected) in test_cases {
            assert_eq!(
                Argument::parse(input).unwrap(),
                expected,
                "failed to parse valid argument: {input:?}"
            );
        }
    }

    #[test]
    fn parse_invalid_arguments() {
        let test_cases = [
            // Empty or unknown selection methods.
            "",
            "unknown",
            // Arguments supplied to methods that do not accept them.
            "random=1",
            "proportional=1",
            // Missing arguments.
            "normalized",
            "best",
            "percentile",
            "ranked-linear",
            "ranked-exponential",
            // Invalid numeric arguments.
            "normalized=abc",
            "best=abc",
            "percentile=abc",
            "ranked-linear=abc",
            "ranked-exponential=abc",
            // Non-finite floating-point arguments.
            "normalized=NaN",
            "normalized=inf",
            "normalized=-inf",
            "ranked-linear=NaN",
            "ranked-linear=inf",
            "ranked-linear=-inf",
            // Invalid ranges.
            "percentile=-0.1",
            "percentile=1.1",
            "ranked-linear=-0.1",
            "ranked-linear=1.1",
            // Invalid zero values.
            "best=0",
            "ranked-exponential=0",
        ];

        for input in test_cases {
            assert!(
                Argument::parse(input).is_err(),
                "expected invalid argument to fail: {input:?}"
            );
        }
    }
}
