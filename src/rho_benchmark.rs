#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(clippy::too_many_arguments)]

use std::time::SystemTime;

use nalgebra::{DMatrix, RealField, Scalar, SimdRealField, convert};

use crate::tinympc::{TinyCache, TinyWorkspace};

#[derive(Debug)]
pub struct RhoAdapter<F> {
    pub rho_min: F,
    pub rho_max: F,
    pub clip: bool,
    pub matrices_initialized: bool,

    // Pre-allocated matrices for formatting
    pub A_matrix: DMatrix<F>,
    pub z_vector: DMatrix<F>,
    pub y_vector: DMatrix<F>,
    pub x_decision: DMatrix<F>,
    pub P_matrix: DMatrix<F>,
    pub q_vector: DMatrix<F>,

    // Pre-allocated matrices for residual computation
    pub Ax_vector: DMatrix<F>,
    pub r_prim_vector: DMatrix<F>,
    pub r_dual_vector: DMatrix<F>,
    pub Px_vector: DMatrix<F>,
    pub ATy_vector: DMatrix<F>,

    // Dimensions
    pub format_nx: usize,
    pub format_nu: usize,
    pub format_N: usize,
}

impl<F> Default for RhoAdapter<F>
where
    F: Scalar + Copy + SimdRealField + RealField,
{
    fn default() -> Self {
        Self {
            rho_min: convert(0.0),
            rho_max: convert(0.0),
            clip: true,
            matrices_initialized: false,
            A_matrix: DMatrix::default(),
            z_vector: DMatrix::default(),
            y_vector: DMatrix::default(),
            x_decision: DMatrix::default(),
            P_matrix: DMatrix::default(),
            q_vector: DMatrix::default(),
            Ax_vector: DMatrix::default(),
            r_prim_vector: DMatrix::default(),
            r_dual_vector: DMatrix::default(),
            Px_vector: DMatrix::default(),
            ATy_vector: DMatrix::default(),
            format_nx: 0,
            format_nu: 0,
            format_N: 0,
        }
    }
}

pub struct RhoBenchmarkResult<F> {
    pub time_us: u32,
    pub initial_rho: F,
    pub final_rho: F,
    pub pri_res: F,
    pub dual_res: F,
    pub pri_norm: F,
    pub dual_norm: F,
}

impl<F> Default for RhoBenchmarkResult<F>
where
    F: Scalar + Copy + SimdRealField + RealField,
{
    fn default() -> Self {
        Self {
            time_us: 0,
            initial_rho: convert(0.0),
            final_rho: convert(0.0),
            pri_res: convert(0.0),
            dual_res: convert(0.0),
            pri_norm: convert(0.0),
            dual_norm: convert(0.0),
        }
    }
}

impl<F> RhoAdapter<F>
where
    F: Scalar + Copy + SimdRealField + RealField,
{
    // Initialize matrices for formatting
    pub fn initialize_format_matrices(&mut self, Nx: usize, Nu: usize, N: usize) {
        // Calculate dimensions
        let x_decision_size = Nx * N + Nu * (N - 1);
        let constraint_rows = (Nx + Nu) * (N - 1);

        // Pre-allocate matrices
        self.A_matrix = DMatrix::zeros(constraint_rows, x_decision_size);
        self.z_vector = DMatrix::zeros(constraint_rows, 1);
        self.y_vector = DMatrix::zeros(constraint_rows, 1);
        self.x_decision = DMatrix::zeros(x_decision_size, 1);

        // Pre-compute P matrix structure
        self.P_matrix = DMatrix::zeros(x_decision_size, x_decision_size);
        self.q_vector = DMatrix::zeros(x_decision_size, 1);

        // Pre-allocate residual computation matrices
        self.Ax_vector = DMatrix::zeros(constraint_rows, 1);
        self.r_prim_vector = DMatrix::zeros(constraint_rows, 1);
        self.r_dual_vector = DMatrix::zeros(x_decision_size, 1);
        self.Px_vector = DMatrix::zeros(x_decision_size, 1);
        self.ATy_vector = DMatrix::zeros(x_decision_size, 1);

        // Store dimensions
        self.format_nx = Nx;
        self.format_nu = Nu;
        self.format_N = N;

        self.matrices_initialized = true;
    }

    // Format matrices for residual computation
    pub fn format_matrices(
        &mut self,
        x_prev: &DMatrix<F>,
        u_prev: &DMatrix<F>,
        v_prev: &DMatrix<F>,
        z_prev: &DMatrix<F>,
        g_prev: &DMatrix<F>,
        y_prev: &DMatrix<F>,
        cache: &TinyCache<F>,
        work: &TinyWorkspace<F>,
        N: usize,
    ) {
        if !self.matrices_initialized {
            self.initialize_format_matrices(x_prev.nrows(), u_prev.nrows(), N);
        }

        let Nx = self.format_nx;
        let Nu = self.format_nu;

        // Fill x_decision
        let mut x_idx = 0;
        for i in 0..N {
            self.x_decision
                .view_mut((x_idx, 0), (Nx, 1))
                .set_column(0, &x_prev.column(i));
            x_idx += Nx;
            if i < N - 1 {
                self.x_decision
                    .view_mut((x_idx, 0), (Nu, 1))
                    .set_column(0, &u_prev.column(i));
                x_idx += Nu;
            }
        }

        // Clear A matrix for reuse
        self.A_matrix.fill(convert(0.0));

        // Fill A matrix with dynamics and input constraints
        for i in 0..N - 1 {
            // Input constraints
            let mut row_start = i * Nu;
            let mut col_start = i * (Nx + Nu) + Nx;
            let mut ta = self.A_matrix.view_mut((row_start, col_start), (Nu, Nu));
            for j in 0..Nu {
                ta.set_column(j, &DMatrix::identity(Nu, Nu).column(j));
            }

            // Dynamics constraints
            row_start = (N - 1) * Nu + i * Nx;
            col_start = i * (Nx + Nu);
            for j in 0..Nx {
                self.A_matrix
                    .view_mut((row_start, col_start), (Nx, Nx))
                    .set_column(j, &work.Adyn.column(j));
            }

            for j in 0..Nu {
                self.A_matrix
                    .view_mut((row_start, col_start + Nx), (Nx, Nu))
                    .set_column(j, &work.Bdyn.column(j));
            }

            let next_state_idx = col_start + Nx + Nu;
            if next_state_idx < self.A_matrix.ncols() {
                for j in 0..Nx {
                    self.A_matrix
                        .view_mut((row_start, next_state_idx), (Nx, Nx))
                        .set_column(j, &-DMatrix::identity(Nx, Nx).column(j));
                }
            }
        }

        // Fill z and y vectors
        for i in 0..N - 1 {
            self.z_vector
                .view_mut((i * Nu, 0), (Nu, 1))
                .set_column(0, &z_prev.column(i));
            self.z_vector
                .view_mut(((N - 1) * Nu + i * Nx, 0), (Nx, 1))
                .set_column(0, &v_prev.column(i + 1));

            self.y_vector
                .view_mut((i * Nu, 0), (Nu, 1))
                .set_column(0, &y_prev.column(i));
            self.y_vector
                .view_mut(((N - 1) * Nu + i * Nx, 0), (Nx, 1))
                .set_column(0, &g_prev.column(i + 1));
        }

        // Build P matrix (cost matrix)
        self.P_matrix.fill(convert(0.0));

        // Fill diagonal blocks
        x_idx = 0;
        for i in 0..N {
            // State cost
            if i == N - 1 {
                for j in 0..Nx {
                    self.P_matrix
                        .view_mut((x_idx, x_idx), (Nx, Nx))
                        .set_column(j, &cache.Pinf.column(j));
                }
            } else {
                for j in 0..Nx {
                    self.P_matrix
                        .view_mut((x_idx, x_idx), (Nx, Nx))
                        .set_column(j, &DMatrix::from_diagonal(&work.Q).column(j));
                }
            }
            x_idx += Nx;

            // Input cost
            if i < N - 1 {
                for j in 0..Nu {
                    self.P_matrix
                        .view_mut((x_idx, x_idx), (Nu, Nu))
                        .set_column(j, &DMatrix::from_diagonal(&work.R).column(j));
                }
                x_idx += Nu;
            }
        }

        // Create q vector (linear cost vector)
        x_idx = 0;
        for i in 0..N {
            // For simplicity, we'll use zero reference for now
            // In a real implementation, you'd use your reference trajectory
            let x_ref = DMatrix::zeros(Nx, 1);
            let delta_x = x_prev.column(i) - x_ref;
            self.q_vector
                .view_mut((x_idx, 0), (Nx, 1))
                .set_column(0, &(&DMatrix::from_diagonal(&work.Q) * &delta_x));
            x_idx += Nx;

            if i < N - 1 {
                // For simplicity, we'll use zero reference for now
                let u_ref = DMatrix::zeros(Nu, 1);
                let delta_u = u_prev.column(i) - u_ref;
                self.q_vector
                    .view_mut((x_idx, 0), (Nu, 1))
                    .set_column(0, &(&DMatrix::from_diagonal(&work.R) * &delta_u));
                x_idx += Nu;
            }
        }
    }

    // Compute residuals
    pub fn compute_residuals(
        &mut self,
        pri_res: &mut F,
        dual_res: &mut F,
        pri_norm: &mut F,
        dual_norm: &mut F,
    ) {
        // Compute Ax
        self.Ax_vector = &self.A_matrix * &self.x_decision;

        // Compute primal residual
        self.r_prim_vector = &self.Ax_vector - &self.z_vector;
        *pri_res = self.r_prim_vector.abs().max();
        *pri_norm = self.Ax_vector.abs().max().max(self.z_vector.abs().max());

        // Compute dual residual components
        self.Px_vector = &self.P_matrix * &self.x_decision;
        self.ATy_vector = &self.A_matrix.transpose() * &self.y_vector;

        // Compute full dual residual
        self.r_dual_vector = &self.Px_vector + &self.q_vector + &self.ATy_vector;
        *dual_res = self.r_dual_vector.abs().max();

        // Compute normalization
        *dual_norm = self
            .Px_vector
            .abs()
            .max()
            .max(self.ATy_vector.abs().max())
            .max(self.q_vector.abs().max());
    }

    // Predict new rho value
    pub fn predict_rho(
        &self,
        pri_res: F,
        dual_res: F,
        pri_norm: F,
        dual_norm: F,
        current_rho: F,
    ) -> F {
        const EPS: f64 = 1e-10;

        let normalized_pri = pri_res / (pri_norm + convert(EPS));
        let normalized_dual = dual_res / (dual_norm + convert(EPS));

        let ratio = normalized_pri / (normalized_dual + convert(EPS));

        let mut new_rho = current_rho * ratio.sqrt();

        if self.clip {
            new_rho = new_rho.max(self.rho_min).min(self.rho_max);
        }

        new_rho
    }

    // Update matrices using derivatives
    pub fn update_matrices_with_derivatives(&self, cache: &mut TinyCache<F>, new_rho: F) {
        let delta_rho = new_rho - cache.rho;

        cache.Kinf += cache.dKinf_drho.scale(delta_rho);
        cache.Pinf += cache.dPinf_drho.scale(delta_rho);
        cache.C1 += cache.dC1_drho.scale(delta_rho);
        cache.C2 += cache.dC2_drho.scale(delta_rho);

        cache.rho = new_rho;
    }

    // Main benchmark function
    pub fn benchmark_rho_adaptation(
        &mut self,
        x_prev: &DMatrix<F>,
        u_prev: &DMatrix<F>,
        v_prev: &DMatrix<F>,
        z_prev: &DMatrix<F>,
        g_prev: &DMatrix<F>,
        y_prev: &DMatrix<F>,
        cache: &mut TinyCache<F>,
        work: &TinyWorkspace<F>,
        N: usize,
        result: &mut RhoBenchmarkResult<F>,
    ) {
        let start_time = micros();

        // Format matrices
        self.format_matrices(
            x_prev, u_prev, v_prev, z_prev, g_prev, y_prev, cache, work, N,
        );

        // Compute residuals
        let mut pri_res: F = convert(0.0);
        let mut dual_res: F = convert(0.0);
        let mut pri_norm: F = convert(0.0);
        let mut dual_norm: F = convert(0.0);
        self.compute_residuals(&mut pri_res, &mut dual_res, &mut pri_norm, &mut dual_norm);

        // Predict new rho
        let new_rho = self.predict_rho(pri_res, dual_res, pri_norm, dual_norm, cache.rho);

        // Update matrices
        self.update_matrices_with_derivatives(cache, new_rho);

        // Store results
        result.time_us = micros() - start_time;
        result.initial_rho = cache.rho;
        result.final_rho = new_rho;
        result.pri_res = pri_res;
        result.dual_res = dual_res;
        result.pri_norm = pri_norm;
        result.dual_norm = dual_norm;
    }
}

pub fn micros() -> u32 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .subsec_micros()
}
