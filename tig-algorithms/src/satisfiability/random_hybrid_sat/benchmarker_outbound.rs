/*!
Copyright 2024 Crypti (PTY) LTD

Licensed under the TIG Benchmarker Outbound Game License v1.0 (the "License"); you 
may not use this file except in compliance with the License. You may obtain a copy 
of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::HashMap;
use tig_challenges::satisfiability::*;

pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
    let mut rng = StdRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()) as u64);

    let mut p_single = vec![false; challenge.difficulty.num_variables];
    let mut n_single = vec![false; challenge.difficulty.num_variables];

    let mut clauses_ = challenge.clauses.clone();
    let mut clauses: Vec<Vec<i32>> = Vec::with_capacity(clauses_.len());

    let mut rounds = 0;

    let mut dead = false;

    // PERFORMANCE HOTSPOT 1: This while loop could be optimized
    while !(dead) {
        let mut done = true;
        // PERFORMANCE HOTSPOT 2: This loop could potentially be parallelized
        for c in &clauses_ {
            let mut c_: Vec<i32> = Vec::with_capacity(c.len()); 
            let mut skip = false;
            for (i, l) in c.iter().enumerate() {
                if (p_single[(l.abs() - 1) as usize] && *l > 0)
                    || (n_single[(l.abs() - 1) as usize] && *l < 0)
                    || c[(i + 1)..].contains(&-l)
                {
                    skip = true;
                    break;
                } else if p_single[(l.abs() - 1) as usize]
                    || n_single[(l.abs() - 1) as usize]
                    || c[(i + 1)..].contains(&l)
                {
                    done = false;
                    continue;
                } else {
                    c_.push(*l);
                }
            }
            if skip {
                done = false;
                continue;
            };
            match c_[..] {
                [l] => {
                    done = false;
                    if l > 0 {
                        if n_single[(l.abs() - 1) as usize] {
                            dead = true;
                            break;
                        } else {
                            p_single[(l.abs() - 1) as usize] = true;
                        }
                    } else {
                        if p_single[(l.abs() - 1) as usize] {
                            dead = true;
                            break;
                        } else {
                            n_single[(l.abs() - 1) as usize] = true;
                        }
                    }
                }
                [] => {
                    dead = true;
                    break;
                }
                _ => {
                    clauses.push(c_);
                }
            }
        }
        if done {
            break;
        } else {
            clauses_ = clauses;
            clauses = Vec::with_capacity(clauses_.len());
        }
    }

    if dead {
        return Ok(None);
    }

    let num_variables = challenge.difficulty.num_variables;
    let num_clauses = clauses.len();

    let mut p_clauses: Vec<Vec<usize>> = vec![Vec::new(); num_variables];
    let mut n_clauses: Vec<Vec<usize>> = vec![Vec::new(); num_variables];

    let mut variables = vec![false; num_variables];
    for v in 0..num_variables {
        if p_single[v] {
            variables[v] = true
        } else if n_single[v] {
            variables[v] = false
        } else {
            variables[v] = rng.gen_bool(0.5)
        }
    }
    let mut num_good_so_far: Vec<usize> = vec![0; num_clauses];

    // PERFORMANCE HOTSPOT 3: This loop could be optimized
    for c in &clauses {
        for &l in c {
            let var = (l.abs() - 1) as usize;
            if l > 0 {
                if p_clauses[var].capacity() == 0 {
                    p_clauses[var] = Vec::with_capacity(clauses.len() / num_variables + 1);
                }
            } else {
                if n_clauses[var].capacity() == 0 {
                    n_clauses[var] = Vec::with_capacity(clauses.len() / num_variables + 1);
                }
            }
        }
    }

    // PERFORMANCE HOTSPOT 4: This nested loop is a major bottleneck
    // Pre-allocate capacity for p_clauses and n_clauses
    let estimated_capacity = clauses.len() / num_variables + 1;
    for var in 0..num_variables {
        p_clauses[var].reserve(estimated_capacity);
        n_clauses[var].reserve(estimated_capacity);
    }

    // Use iterators and flatten the nested loop
    clauses.iter().enumerate().flat_map(|(i, c)| {
        c.iter().map(move |&l| (i, l))
    }).for_each(|(i, l)| {
        let var = (l.abs() - 1) as usize;
        if l > 0 {
            p_clauses[var].push(i);
            num_good_so_far[i] += variables[var] as usize;
        } else {
            n_clauses[var].push(i);
            num_good_so_far[i] += (!variables[var]) as usize;
        }
    });

    let mut residual_ = Vec::with_capacity(num_clauses);
    let mut residual_indices = HashMap::with_capacity(num_clauses);

    // PERFORMANCE HOTSPOT 5: This loop could be optimized or parallelized
    for (i, &num_good) in num_good_so_far.iter().enumerate() {
        if num_good == 0 {
            residual_.push(i);
            residual_indices.insert(i, residual_.len() - 1);
        }
    }

    loop {
        if !residual_.is_empty() {
            
            let i = residual_[rng.gen_range(0..residual_.len())];
            let mut min_sad = clauses.len();
            let mut v_min_sad = Vec::with_capacity(clauses[i].len()); 
            let c = &clauses[i];
            // PERFORMANCE HOTSPOT 6: This nested loop is another major bottleneck
            for &l in c {
                let mut sad = 0 as usize;
                if variables[(l.abs() - 1) as usize] {
                    for &c in &p_clauses[(l.abs() - 1) as usize] {
                        if num_good_so_far[c] == 1 {
                            sad += 1;
                            if sad > min_sad {
                                break;
                            }
                        }
                    }
                } else {
                    for &c in &n_clauses[(l.abs() - 1) as usize] {
                        if num_good_so_far[c] == 1 {
                            sad += 1;
                            if sad > min_sad {
                                break;
                            }
                        }
                    }
                }

                if sad < min_sad {
                    min_sad = sad;
                    v_min_sad.clear();
                    v_min_sad.push((l.abs() - 1) as usize);
                } else if sad == min_sad {
                    v_min_sad.push((l.abs() - 1) as usize);
                }
            }
            let v = if min_sad == 0 {
                if v_min_sad.len() == 1 {
                    v_min_sad[0]
                } else {
                    v_min_sad[rng.gen_range(0..(v_min_sad.len() as u32)) as usize]
                }
            } else {
                if rng.gen_bool(0.5) {
                    let l = c[rng.gen_range(0..(c.len() as u32)) as usize];
                    (l.abs() - 1) as usize
                } else {
                    v_min_sad[rng.gen_range(0..(v_min_sad.len() as u32)) as usize]
                }
            };

            if variables[v] {
                for &c in &n_clauses[v] {
                    num_good_so_far[c] += 1;
                    if num_good_so_far[c] == 1 {
                        let i = residual_indices.remove(&c).unwrap();
                        let last = residual_.pop().unwrap();
                        if i < residual_.len() {
                            residual_[i] = last;
                            residual_indices.insert(last, i);
                        }
                    }
                }
                for &c in &p_clauses[v] {
                    if num_good_so_far[c] == 1 {
                        residual_.push(c);
                        residual_indices.insert(c, residual_.len() - 1);
                    }
                    num_good_so_far[c] -= 1;
                }
            } else {
                for &c in &n_clauses[v] {
                    if num_good_so_far[c] == 1 {
                        residual_.push(c);
                        residual_indices.insert(c, residual_.len() - 1);
                    }
                    num_good_so_far[c] -= 1;
                }

                for &c in &p_clauses[v] {
                    num_good_so_far[c] += 1;
                    if num_good_so_far[c] == 1 {
                        let i = residual_indices.remove(&c).unwrap();
                        let last = residual_.pop().unwrap();
                        if i < residual_.len() {
                            residual_[i] = last;
                            residual_indices.insert(last, i);
                        }
                    }
                }
            }

            variables[v] = !variables[v];
        } else {
            break;
        }
        rounds += 1;
        // PERFORMANCE HOTSPOT 7: This main loop is a critical area for optimization
        if rounds >= num_variables * 35 {
            return Ok(None);
        }
    }

    return Ok(Some(Solution { variables }));
}

#[cfg(feature = "cuda")]
mod gpu_optimisation {
    use super::*;
    use cudarc::driver::*;
    use std::{collections::HashMap, sync::Arc};
    use tig_challenges::CudaKernel;

    // set KERNEL to None if algorithm only has a CPU implementation
    pub const KERNEL: Option<CudaKernel> = None;

    // Important! your GPU and CPU version of the algorithm should return the same result
    pub fn cuda_solve_challenge(
        challenge: &Challenge,
        dev: &Arc<CudaDevice>,
        mut funcs: HashMap<&'static str, CudaFunction>,
    ) -> anyhow::Result<Option<Solution>> {
        solve_challenge(challenge)
    }
}
#[cfg(feature = "cuda")]
pub use gpu_optimisation::{cuda_solve_challenge, KERNEL};