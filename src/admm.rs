use log::debug;
use nalgebra::{RealField, Scalar, SimdRealField};

use crate::{
    TinySolver,
    rho_benchmark::{RhoAdapter, RhoBenchmarkResult},
};

impl<F> TinySolver<F>
where
    F: Scalar + Copy + SimdRealField + RealField,
{
    /// Update linear terms from Riccati backward pass
    pub(crate) fn backward_pass_grad(&mut self) {
        for i in (0..self.work.N - 1).rev() {
            self.work.d.set_column(
                i,
                &(&self.cache.Quu_inv
                    * (self.work.Bdyn.transpose() * self.work.p.column(i + 1)
                        + self.work.r.column(i))),
            );
            self.work.p.set_column(
                i,
                &(self.work.q.column(i) + &self.cache.AmBKt * self.work.p.column(i + 1)
                    - (self.cache.Kinf.transpose() * self.work.r.column(i))),
            );
        }
        // debug!("d {}", self.work.d);
        // debug!("p {}", self.work.p);
    }

    /// Use LQR feedback policy to roll out trajectory
    pub(crate) fn forward_pass(&mut self) {
        // Forward-pass for rest of horizon
        for i in 0..self.work.N - 1 {
            self.work.u.set_column(
                i,
                &(-&self.cache.Kinf * self.work.x.column(i) - self.work.d.column(i)),
            );
            self.work.x.set_column(
                i + 1,
                &(&self.work.Adyn * self.work.x.column(i)
                    + &self.work.Bdyn * self.work.u.column(i)),
            );
        }
    }

    /// Project slack (auxiliary) variables into their feasible domain, defined by projection functions related to each constraint
    /// TODO: pass in meta information with each constraint assigning it to a projection function
    pub(crate) fn update_slack(&mut self) {
        self.work.znew = &self.work.u + &self.work.y;
        self.work.vnew = &self.work.x + &self.work.g;

        // Box constraints on input
        if self.settings.en_input_bound {
            self.work
                .znew
                .zip_zip_apply(&self.work.u_min, &self.work.u_max, |u, min, max| {
                    *u = (*u).clamp(min, max)
                });
        }

        // Box constraints on state
        if self.settings.en_state_bound {
            self.work
                .vnew
                .zip_zip_apply(&self.work.x_min, &self.work.x_max, |x, min, max| {
                    *x = (*x).clamp(min, max)
                });
        }
    }

    /// Update next iteration of dual variables by performing the augmented lagrangian multiplier update
    pub(crate) fn update_dual(&mut self) {
        // Gadient ascent
        self.work.y = &self.work.y + &self.work.u - &self.work.znew;
        self.work.g = &self.work.g + &self.work.x - &self.work.vnew;
    }

    /// Update linear control cost terms in the Riccati feedback using the changing slack and dual variables from ADMM
    pub(crate) fn update_linear_cost(&mut self) {
        // r
        self.work.Uref.column_iter().enumerate().for_each(|(i, u)| {
            self.work.r.set_column(i, &(-u.component_mul(&self.work.R)));
        });
        self.work.r -= (&self.work.znew - &self.work.y).scale(self.cache.rho);

        // q
        self.work.Xref.column_iter().enumerate().for_each(|(i, x)| {
            self.work.q.set_column(i, &(-x.component_mul(&self.work.Q)));
        });
        self.work.q -= (&self.work.vnew - &self.work.g).scale(self.cache.rho);

        // p
        let p =
            -(self.work.Xref.column(self.work.N - 1).transpose() * &self.cache.Pinf).transpose();
        let p = p
            - (self.work.vnew.column(self.work.N - 1) - self.work.g.column(self.work.N - 1))
                .scale(self.cache.rho);
        self.work.p.set_column(self.work.N - 1, &p);
    }

    /// Check for termination condition by evaluating whether the largest absolute primal and dual residuals for states and inputs are below threhold.
    pub(crate) fn termination_condition(&mut self) -> bool {
        if self.work.iter.is_multiple_of(self.settings.check_termination) {
            // Calculate residuals on slack variables
            self.work.primal_residual_state = (&self.work.x - &self.work.vnew).abs().max();
            self.work.dual_residual_state =
                (&self.work.v - &self.work.vnew).abs().max() * self.cache.rho;
            self.work.primal_residual_input = (&self.work.u - &self.work.znew).abs().max();
            self.work.dual_residual_input =
                (&self.work.z - &self.work.znew).abs().max() * self.cache.rho;

            // If all residuals are below tolerance, we terminate
            self.work.primal_residual_state < self.settings.abs_pri_tol
                && self.work.primal_residual_input < self.settings.abs_pri_tol
                && self.work.dual_residual_state < self.settings.abs_dua_tol
                && self.work.dual_residual_input < self.settings.abs_dua_tol
        } else {
            false
        }
    }

    pub(crate) fn admm_solve(&mut self) -> bool {
        // Initialize variables
        self.solution.solved = 0;
        self.solution.iter = 0;
        self.work.status = 11; // TINY_UNSOLVED
        self.work.iter = 0;

        // Setup for adaptive rho
        let mut adapter = RhoAdapter {
            rho_min: self.settings.adaptive_rho_min,
            rho_max: self.settings.adaptive_rho_max,
            clip: self.settings.adaptive_rho_enable_clipping,
            ..Default::default()
        };

        let mut rho_result = RhoBenchmarkResult::default();

        // Store previous values for residuals
        let mut v_prev = self.work.vnew.clone();
        let mut z_prev = self.work.znew.clone();

        for i in 0..self.settings.max_iter {
            // Solve linear system with Riccati and roll out to get new trajectory
            self.forward_pass();

            // Project slack variables into feasible domain
            self.update_slack();

            // Compute next iteration of dual variables
            self.update_dual();

            // Update linear control cost terms using reference trajectory, duals, and slack variables
            self.update_linear_cost();

            self.work.iter += 1;

            if self.settings.adaptive_rho {
                // Calculate residuals for adaptive rho
                let _pri_res_input = (&self.work.u - &self.work.znew).abs().max();
                let _pri_res_state = (&self.work.x - &self.work.vnew).abs().max();
                let _dua_res_input = self.cache.rho * (&self.work.znew - &z_prev).abs().max();
                let _dua_res_state = self.cache.rho * (&self.work.vnew - &v_prev).abs().max();

                // Update rho every 5 iterations
                if i > 0 && i % 5 == 0 {
                    adapter.benchmark_rho_adaptation(
                        &self.work.x,
                        &self.work.u,
                        &self.work.vnew,
                        &self.work.znew,
                        &self.work.g,
                        &self.work.y,
                        &mut self.cache,
                        &self.work,
                        self.work.N,
                        &mut rho_result,
                    );

                    // Update matrices using Taylor expansion
                    adapter.update_matrices_with_derivatives(&mut self.cache, rho_result.final_rho);
                }
            }

            // Store previous values for next iteration
            z_prev = self.work.znew.clone();
            v_prev = self.work.vnew.clone();

            // Check for whether cost is minimized by calculating residuals
            if self.termination_condition() {
                self.work.status = 1; // TINY_SOLVED

                // Save solution
                self.solution.iter = self.work.iter;
                self.solution.solved = 1;
                self.solution.x = self.work.vnew.clone();
                self.solution.u = self.work.znew.clone();

                debug!("Solver converged in {} iterations", self.work.iter);
                return true;
            }

            // Save previous slack variables
            self.work.v = self.work.vnew.clone();
            self.work.z = self.work.znew.clone();

            self.backward_pass_grad();
        }

        self.solution.iter = self.work.iter;
        self.solution.solved = 0;
        self.solution.x = self.work.vnew.clone();
        self.solution.u = self.work.znew.clone();
        false
    }
}
