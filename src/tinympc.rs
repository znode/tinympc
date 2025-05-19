#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(unused)]

// Rust port of TinyMPC["https://github.com/TinyMPC/TinyMPC"]

use log::{debug, warn};
use nalgebra::{
    DMatrix, DVector, RealField, SMatrix, SVector, Scalar, SimdRealField, SimdValue, convert,
    matrix, zero,
};

// Default settings
const TINY_DEFAULT_ABS_PRI_TOL: f64 = 1e-03;
const TINY_DEFAULT_ABS_DUA_TOL: f64 = 1e-03;
const TINY_DEFAULT_MAX_ITER: usize = 1000;
const TINY_DEFAULT_CHECK_TERMINATION: usize = 1;
const TINY_DEFAULT_EN_STATE_BOUND: bool = true;
const TINY_DEFAULT_EN_INPUT_BOUND: bool = true;

#[derive(Debug)]
pub struct TinySolver<F> {
    pub solution: TinySolution<F>, // Solution
    pub settings: TinySettings<F>, // Problem settings
    pub cache: TinyCache<F>,       // Problem cache
    pub work: TinyWorkspace<F>,    // Solver workspace
}

/// Solution
#[derive(Debug)]
pub struct TinySolution<F> {
    pub iter: usize,
    pub solved: usize,
    pub x: DMatrix<F>, // Nx * N
    pub u: DMatrix<F>, // Nu * N-1
}

/// Matrices that must be recomputed with changes in time step, rho
#[derive(Debug)]
pub struct TinyCache<F> {
    pub rho: F,
    /// Infinite-time horizon LQR gain
    pub Kinf: DMatrix<F>, // Nu * Nx

    /// Infinite-time horizon LQR Hessian
    pub Pinf: DMatrix<F>, // Nx * Nx

    ///
    pub Quu_inv: DMatrix<F>, // Nu * Nu

    /// Precomputed `(A - B * Klqr)^T`
    pub AmBKt: DMatrix<F>, // Nx * Nx

    pub C1: DMatrix<F>, // Nu * Nu
    pub C2: DMatrix<F>, // Nx * Nx

    /// Add sensitivity matrices for Taylor updates
    pub dKinf_drho: DMatrix<F>, // Nu * Nx
    pub dPinf_drho: DMatrix<F>, // Nx * Nx
    pub dC1_drho: DMatrix<F>,   // Nu * Nu
    pub dC2_drho: DMatrix<F>,   // Nx * Nx
}

impl<F> TinyCache<F>
where
    F: Scalar + Copy + SimdRealField + RealField,
{
    pub fn new(
        Adyn: DMatrix<F>, // Nx * Nx
        Bdyn: DMatrix<F>, // Nx * Nu
        Q: DMatrix<F>,    // Nx * Nx
        R: DMatrix<F>,    // Nu * Nu
        Nx: usize,
        Nu: usize,
        rho: F,
    ) -> Self {
        // Update by adding rho * identity matrix to Q, R
        let Q1 = Q + DMatrix::<F>::identity(Nx, Nx).scale(rho);
        let R1 = R + DMatrix::<F>::identity(Nu, Nu).scale(rho);

        // Printing
        debug!("A = {}", Adyn);
        debug!("B = {}", Bdyn);
        debug!("Q = {}", Q1);
        debug!("R = {}", R1);
        debug!("rho = {} ", rho);

        // Riccati recursion to get Kinf, Pinf
        let mut Ktp1 = DMatrix::zeros(Nu, Nx);
        let mut Ptp1 = DMatrix::zeros(Nx, Nx);
        Ptp1.fill_diagonal(convert(1.0));
        Ptp1.scale(rho);
        let mut Kinf = DMatrix::zeros(Nu, Nx);
        let mut Pinf = DMatrix::zeros(Nx, Nx);

        let At = Adyn.clone().transpose();
        let Bt = Bdyn.clone().transpose();

        for i in 0..1000 {
            Kinf = (&R1 + &Bt * &Ptp1 * &Bdyn).try_inverse().unwrap() * &Bt * &Ptp1 * &Adyn;

            Pinf = &Q1 + &At * &Ptp1 * (&Adyn - &Bdyn * &Kinf);
            // if Kinf converges, break
            if ((&Kinf - &Ktp1).abs().max() < convert(1e-5)) {
                debug!("Kinf converged after {} iterations", i + 1);
                break;
            }
            Ktp1 = Kinf.clone();
            Ptp1 = Pinf.clone();
        }

        // Compute cached matrices
        let Quu_inv = (&R1 + &Bt * &Pinf * &Bdyn).try_inverse().unwrap();
        let AmBKt = (&Adyn - &Bdyn * &Kinf).transpose();

        debug!("Kinf = {}", Kinf);
        debug!("Pinf = {}", Pinf);
        debug!("Quu_inv = {}", Quu_inv);
        debug!("AmBKt = {}", AmBKt);
        debug!("Precomputation finished!");

        Self {
            rho,
            Kinf,
            Pinf,
            Quu_inv: Quu_inv.clone(),
            AmBKt: AmBKt.clone(),
            C1: Quu_inv,
            C2: AmBKt,
            dKinf_drho: DMatrix::zeros(Nu, Nx),
            dPinf_drho: DMatrix::zeros(Nx, Nx),
            dC1_drho: DMatrix::zeros(Nu, Nu),
            dC2_drho: DMatrix::zeros(Nx, Nx),
        }
    }

    pub fn initialize_sensitivity_matrices(&mut self, nx: usize, nu: usize) {
        // Initialize matrices with zeros
        self.dKinf_drho = DMatrix::zeros(nu, nx);
        self.dPinf_drho = DMatrix::zeros(nx, nx);
        self.dC1_drho = DMatrix::zeros(nu, nu);
        self.dC2_drho = DMatrix::zeros(nx, nx);

        #[rustfmt::skip]
        const dKinf_drho: [f64;48 /*4 x 12*/] = [
              0.0001,  -0.0001,  -0.0025,   0.0003,   0.0007,   0.0050,   0.0001,  -0.0001,  -0.0008,   0.0000,   0.0001,   0.0008,
             -0.0001,  -0.0000,  -0.0025,  -0.0001,  -0.0006,  -0.0050,  -0.0001,   0.0000,  -0.0008,  -0.0000,  -0.0001,  -0.0008,
              0.0000,   0.0000,  -0.0025,   0.0001,   0.0004,   0.0050,   0.0000,   0.0000,  -0.0008,   0.0000,   0.0000,   0.0008,
             -0.0000,   0.0001,  -0.0025,  -0.0003,  -0.0004,  -0.0050,  -0.0000,   0.0001,  -0.0008,  -0.0000,  -0.0000,  -0.0008
        ];

        #[rustfmt::skip]
        const dPinf_drho: [f64;144 /*12 x 12*/] = [
             0.0494,  -0.0045,  -0.0000,   0.0110,   0.1300,  -0.0283,   0.0280,  -0.0026,  -0.0000,   0.0004,   0.0070,  -0.0094,
            -0.0045,   0.0491,   0.0000,  -0.1320,  -0.0111,   0.0114,  -0.0026,   0.0279,   0.0000,  -0.0076,  -0.0004,   0.0038,
            -0.0000,   0.0000,   2.4450,   0.0000,  -0.0000,  -0.0000,  -0.0000,   0.0000,   1.2593,   0.0000,   0.0000,   0.0000,
             0.0110,  -0.1320,   0.0000,   0.3913,   0.0592,   0.3108,   0.0080,  -0.0776,   0.0000,   0.0254,   0.0068,   0.0750,
             0.1300,  -0.0111,  -0.0000,   0.0592,   0.4420,   0.7771,   0.0797,  -0.0081,  -0.0000,   0.0068,   0.0350,   0.1875,
            -0.0283,   0.0114,  -0.0000,   0.3108,   0.7771,  10.0441,   0.0272,  -0.0109,   0.0000,   0.0655,   0.1639,   2.6362,
             0.0280,  -0.0026,  -0.0000,   0.0080,   0.0797,   0.0272,   0.0163,  -0.0016,  -0.0000,   0.0005,   0.0047,   0.0032,
            -0.0026,   0.0279,   0.0000,  -0.0776,  -0.0081,  -0.0109,  -0.0016,   0.0161,   0.0000,  -0.0046,  -0.0005,  -0.0013,
            -0.0000,   0.0000,   1.2593,   0.0000,  -0.0000,   0.0000,  -0.0000,   0.0000,   0.9232,   0.0000,   0.0000,   0.0000,
             0.0004,  -0.0076,   0.0000,   0.0254,   0.0068,   0.0655,   0.0005,  -0.0046,   0.0000,   0.0022,   0.0017,   0.0244,
             0.0070,  -0.0004,   0.0000,   0.0068,   0.0350,   0.1639,   0.0047,  -0.0005,   0.0000,   0.0017,   0.0054,   0.0610,
            -0.0094,   0.0038,   0.0000,   0.0750,   0.1875,   2.6362,   0.0032,  -0.0013,   0.0000,   0.0244,   0.0610,   0.9869
        ];

        #[rustfmt::skip]
        const dC1_drho: [f64; 16 /*4 x 4*/] = [
             -0.0000,   0.0000,  -0.0000,   0.0000,
              0.0000,  -0.0000,   0.0000,  -0.0000,
             -0.0000,   0.0000,  -0.0000,   0.0000,
              0.0000,  -0.0000,   0.0000,  -0.0000
        ];

        #[rustfmt::skip]
        const dC2_drho: [f64;144 /*12 x 12*/] = [
              0.0000,  -0.0000,   0.0000,   0.0000,   0.0000,  -0.0000,   0.0000,  -0.0000,   0.0000,   0.0000,   0.0000,  -0.0000,
             -0.0000,   0.0000,   0.0000,  -0.0000,  -0.0000,   0.0000,  -0.0000,   0.0000,   0.0000,  -0.0000,  -0.0000,   0.0000,
             -0.0000,   0.0000,   0.0001,   0.0000,  -0.0000,  -0.0000,  -0.0000,   0.0000,   0.0000,   0.0000,  -0.0000,  -0.0000,
              0.0000,  -0.0000,  -0.0000,   0.0001,   0.0000,  -0.0000,   0.0000,  -0.0000,  -0.0000,   0.0000,   0.0000,  -0.0000,
              0.0000,  -0.0000,  -0.0000,   0.0000,   0.0001,  -0.0000,   0.0000,  -0.0000,  -0.0000,   0.0000,   0.0000,  -0.0000,
             -0.0000,   0.0000,  -0.0000,  -0.0000,   0.0000,   0.0001,  -0.0000,   0.0000,  -0.0000,   0.0000,   0.0000,   0.0000,
              0.0000,  -0.0000,   0.0000,   0.0000,   0.0000,  -0.0000,   0.0000,  -0.0000,   0.0000,   0.0000,   0.0000,  -0.0000,
             -0.0000,   0.0000,   0.0000,  -0.0000,  -0.0000,   0.0000,  -0.0000,   0.0000,   0.0000,  -0.0000,  -0.0000,   0.0000,
             -0.0000,   0.0000,   0.0021,   0.0000,  -0.0000,  -0.0000,  -0.0000,   0.0000,   0.0006,   0.0000,  -0.0000,  -0.0000,
              0.0002,  -0.0027,  -0.0000,   0.0068,   0.0005,  -0.0005,   0.0001,  -0.0015,  -0.0000,   0.0004,   0.0000,  -0.0001,
              0.0027,  -0.0002,   0.0000,   0.0005,   0.0066,  -0.0011,   0.0015,  -0.0001,   0.0000,   0.0000,   0.0004,  -0.0002,
             -0.0001,   0.0001,   0.0000,  -0.0000,   0.0000,   0.0041,  -0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0006
        ];

        // Map arrays to Eigen matrices
        // Map<const Matrix<float, 4, 12>>(dKinf_drho[0]).cast<tinytype>();
        self.dKinf_drho =
            DMatrix::from_vec(12, 4, dKinf_drho.iter().map(|x| convert(*x)).collect()).transpose();
        self.dPinf_drho =
            DMatrix::from_vec(12, 12, dPinf_drho.iter().map(|x| convert(*x)).collect()).transpose();
        self.dC1_drho =
            DMatrix::from_vec(4, 4, dC1_drho.iter().map(|x| convert(*x)).collect()).transpose();
        self.dC2_drho =
            DMatrix::from_vec(12, 12, dC2_drho.iter().map(|x| convert(*x)).collect()).transpose();
    }
}

/// User settings
#[derive(Debug)]
pub struct TinySettings<F> {
    pub abs_pri_tol: F,
    pub abs_dua_tol: F,
    pub max_iter: usize,
    pub check_termination: usize,
    pub en_state_bound: bool,
    pub en_input_bound: bool,

    //Add adaptive rho parameters
    pub adaptive_rho: bool,  // Enable/disable adaptive rho (1/0)
    pub adaptive_rho_min: F, // Minimum value for rho
    pub adaptive_rho_max: F, // Maximum value for rho
    pub adaptive_rho_enable_clipping: bool, // Enable/disable clipping of rho (1/0)
}

impl<F> Default for TinySettings<F>
where
    F: Scalar + Copy + SimdRealField + RealField,
{
    fn default() -> Self {
        Self {
            abs_pri_tol: convert(TINY_DEFAULT_ABS_PRI_TOL),
            abs_dua_tol: convert(TINY_DEFAULT_ABS_DUA_TOL),
            max_iter: TINY_DEFAULT_MAX_ITER,
            check_termination: TINY_DEFAULT_CHECK_TERMINATION,
            en_state_bound: TINY_DEFAULT_EN_STATE_BOUND,
            en_input_bound: TINY_DEFAULT_EN_INPUT_BOUND,

            // Default adaptive tho settings
            adaptive_rho: false,
            adaptive_rho_min: convert(1.0),   // Minimum rho value
            adaptive_rho_max: convert(100.0), // Maximum rho value
            adaptive_rho_enable_clipping: true,
        }
    }
}

impl<F> TinySettings<F> {
    pub fn update(
        &mut self,
        abs_pri_tol: F,
        abs_dua_tol: F,
        max_iter: usize,
        check_termination: usize,
        en_state_bound: bool,
        en_input_bound: bool,
    ) {
        self.abs_pri_tol = abs_pri_tol;
        self.abs_dua_tol = abs_dua_tol;
        self.max_iter = max_iter;
        self.check_termination = check_termination;
        self.en_state_bound = en_state_bound;
        self.en_input_bound = en_input_bound;
    }
}

/// Problem variables
#[derive(Debug)]
pub struct TinyWorkspace<F> {
    pub(crate) Nx: usize, // Number of states
    pub(crate) Nu: usize, // Number of control inputs
    pub(crate) N: usize,  // Number of knotpoints in the horizon

    // State and inputs
    pub(crate) x: DMatrix<F>, // Nx * N
    pub(crate) u: DMatrix<F>, // Nu * N-1

    // Linear cost matrices
    pub(crate) q: DMatrix<F>, // Nx * N
    pub(crate) r: DMatrix<F>, // Nu * N-1

    // Riccati backward pass terms
    pub(crate) p: DMatrix<F>, // Nx * N
    pub(crate) d: DMatrix<F>, // Nu * N-1

    // Auxiliary variables
    pub(crate) v: DMatrix<F>,    // Nx * N
    pub(crate) vnew: DMatrix<F>, // Nx * N
    pub(crate) z: DMatrix<F>,    // Nu x N-1
    pub(crate) znew: DMatrix<F>, // Nu x N-1

    // Dual variables
    pub(crate) g: DMatrix<F>, // Nx * N
    pub(crate) y: DMatrix<F>, // Nu * N-1

    // Q, R, A, B given by user
    pub(crate) Q: DVector<F>,    // Nx
    pub(crate) R: DVector<F>,    // Nu
    pub(crate) Adyn: DMatrix<F>, // Nx * Nx
    pub(crate) Bdyn: DMatrix<F>, // Nx * Nu

    // State and input bounds
    pub(crate) x_min: DMatrix<F>, // Nx * N
    pub(crate) x_max: DMatrix<F>, // Nx * N
    pub(crate) u_min: DMatrix<F>, // Nu * N-1
    pub(crate) u_max: DMatrix<F>, // Nu * N-1

    // Reference trajectory to track for one horizon
    pub(crate) Xref: DMatrix<F>, // Nx * N
    pub(crate) Uref: DMatrix<F>, // Nu * N-1

    /// Temporaries
    pub(crate) Qu: DVector<F>, // Nu

    /// Variables for keeping track of solve status
    pub(crate) primal_residual_state: F,
    pub(crate) primal_residual_input: F,
    pub(crate) dual_residual_state: F,
    pub(crate) dual_residual_input: F,
    pub(crate) status: usize,
    pub(crate) iter: usize,
}

impl<F> TinySolver<F>
where
    F: Scalar + Copy + SimdRealField + RealField,
{
    /// Creates a new [`TinySolver<Nx, Nu, Nz, N, F>`].
    ///
    /// ## Arguments
    /// - `A`: State space propagation matrix
    /// - `B`: State space input matrix
    /// - `Q`: State penalty vector
    /// - `R`: Input penalty vector
    ///
    /// Important note about `C`
    #[must_use]
    pub fn new(
        Adyn: DMatrix<F>, // Nx * Nx
        Bdyn: DMatrix<F>, // Nx * Nu
        Q: DMatrix<F>,    // Nx * Nx
        R: DMatrix<F>,    // Nu * Nu
        rho: F,
        Nx: usize,
        Nu: usize,
        N: usize,
        x_min: DMatrix<F>, // Nx * N
        x_max: DMatrix<F>, // Nx * N
        u_min: DMatrix<F>, // Nu * N-1
        u_max: DMatrix<F>, // Nu * N-1
    ) -> Self {
        // Initialize solution
        let solution = TinySolution::<F> {
            iter: 0,
            solved: 0,
            x: DMatrix::zeros(Nx, N),
            u: DMatrix::zeros(Nu, N - 1),
        };

        let settings = TinySettings::default();
        let work = TinyWorkspace::<F> {
            Nx,
            Nu,
            N,
            x: DMatrix::zeros(Nx, N),
            u: DMatrix::zeros(Nu, N - 1),
            q: DMatrix::zeros(Nx, N),
            r: DMatrix::zeros(Nu, N - 1),
            p: DMatrix::zeros(Nx, N),
            d: DMatrix::zeros(Nu, N - 1),
            v: DMatrix::zeros(Nx, N),
            vnew: DMatrix::zeros(Nx, N),
            z: DMatrix::zeros(Nu, N - 1),
            znew: DMatrix::zeros(Nu, N - 1),
            g: DMatrix::zeros(Nx, N),
            y: DMatrix::zeros(Nu, N - 1),
            Q: (Q + DMatrix::identity(Nx, Nx).scale(rho)).diagonal(),
            R: (R + DMatrix::identity(Nu, Nu).scale(rho)).diagonal(),
            Adyn: Adyn.clone(),
            Bdyn: Bdyn.clone(),
            x_min,
            x_max,
            u_min,
            u_max,
            Xref: DMatrix::zeros(Nx, N),
            Uref: DMatrix::zeros(Nu, N - 1),
            Qu: DVector::zeros(Nu),
            primal_residual_state: convert(0.0),
            primal_residual_input: convert(0.0),
            dual_residual_state: convert(0.0),
            dual_residual_input: convert(0.0),
            status: 0,
            iter: 0,
        };

        let mut cache = TinyCache::new(
            Adyn,
            Bdyn,
            DMatrix::from_diagonal(&work.Q),
            DMatrix::from_diagonal(&work.R),
            Nx,
            Nu,
            rho,
        );
        if settings.adaptive_rho {
            cache.initialize_sensitivity_matrices(Nx, Nu);
        }

        Self {
            solution,
            settings,
            cache,
            work,
        }
    }

    /// # Solve for the optimal MPC solution
    ///
    /// This function contains the iterative solver which approximates the minimizing control input `u`.
    ///
    /// Pass the initial condition state `x` , which may be determined using eg. a Kalman filter or observer.
    ///
    /// The function returns;
    /// - the reason for termination, either being convergence or maximum number of iterations
    /// - the optimal actuation `u` to apply to the system
    ///
    pub fn solve(&mut self) -> bool {
        self.admm_solve()
    }

    pub fn set_x0(&mut self, x0: DVector<F>) {
        self.work.x.set_column(0, &x0);
    }

    pub fn set_x_ref(&mut self, x_ref: DMatrix<F>) {
        self.work.Xref = x_ref;
    }

    pub fn set_u_ref(&mut self, u_ref: DMatrix<F>) {
        self.work.Uref = u_ref;
    }

    pub fn get_num_iters(&self) -> usize {
        self.work.iter
    }

    /// Get the system state `x` for the time `o`
    pub fn get_x_at(&self, i: usize) -> DVector<F> {
        self.work.x.column(i).into()
    }

    /// Get the system input `u` for the time `o`
    pub fn get_u_at(&self, i: usize) -> DVector<F> {
        self.work.u.column(i).into()
    }

    /// Get the system input `u` for the current time
    pub fn get_u(&self) -> DVector<F> {
        self.get_u_at(0)
    }

    /// Get reference to matrix containing state predictions
    pub fn get_x_matrix(&self) -> &DMatrix<F> {
        &self.work.x
    }

    /// Get reference to matrix containing input predictions
    pub fn get_u_matrix(&self) -> &DMatrix<F> {
        &self.work.u
    }
}
