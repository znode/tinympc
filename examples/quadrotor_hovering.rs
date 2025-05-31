/// Quadrotor hovering example
///
/// This script is just to show how to use the library, the data for this example is not tuned for our Crazyflie demo. Check the firmware code for more details.
///
///     - NSTATES = 12
///     - NINPUTS = 4
///     - NHORIZON = anything you want
///     States: x (m), y, z, phi, theta, psi, dx, dy, dz, dphi, dtheta, dpsi
///     phi, theta, psi are NOT Euler angles, they are Rodiguez parameters
///     check this paper for more details: https://ieeexplore.ieee.org/document/9326337
///     Inputs: u1, u2, u3, u4 (motor thrust 0-1, order from Crazyflie)
///
use log::debug;
use nalgebra::{DMatrix, DVector, vector};
use tinympc::TinySolver;

const NU: usize = 4;
const NX: usize = 12;
const NH: usize = 10;

fn main() {
    env_logger::init();
    let a = DMatrix::from_vec(NX, NX, A.to_vec()).transpose();
    let b = DMatrix::from_vec(NU, NX, B.to_vec()).transpose();
    let q = DMatrix::from_diagonal(&DVector::from_vec(Q.to_vec()));
    let r = DMatrix::from_diagonal(&DVector::from_vec(R.to_vec()));

    let mut mpc = TinySolver::new(
        a.clone(),
        b.clone(),
        q,
        r,
        RHO,
        NX,
        NU,
        NH,
        DMatrix::from_element(NX, NH, -5.0),
        DMatrix::from_element(NX, NH, 5.0),
        DMatrix::from_element(NU, NH - 1, -0.5),
        DMatrix::from_element(NU, NH - 1, 0.5),
    );

    // Configure settings
    mpc.settings.max_iter = 100;

    // Constant reference through entire horizon
    let reference = vector![0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let mut xref = DMatrix::<f64>::zeros(12, NH);

    for i in 0..NH {
        xref.set_column(i, &reference);
    }

    mpc.set_x_ref(xref.clone());

    // Dynamic state vector
    let mut x = DVector::from_vec(vec![
        0.0, 1.0, 0.0, 0.2, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]);

    let mut total_iters = 0;
    for k in 0..500 {
        debug!(
            "At step {k:3} in {:4} iterations, got tracking error : {:05.4}",
            mpc.num_iters(),
            (&x - reference).norm(),
        );

        // 1、 Update measurement
        mpc.set_x0(x.clone());

        // 2、 Solve MPC problem
        mpc.solve();

        // Iterate simulation
        let u = mpc.u();
        x = &a * x + &b * &u;

        total_iters += mpc.num_iters();
    }

    println!("Total iterations: {total_iters}");
}

#[rustfmt::skip]
const A : [f64;144] = [
    1.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0245250, 0.0000000, 0.0500000, 0.0000000, 0.0000000, 0.0000000, 0.0002044, 0.0000000,
    0.0000000, 1.0000000, 0.0000000, -0.0245250, 0.0000000, 0.0000000, 0.0000000, 0.0500000, 0.0000000, -0.0002044, 0.0000000, 0.0000000,
    0.0000000, 0.0000000, 1.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0500000, 0.0000000, 0.0000000, 0.0000000,
    0.0000000, 0.0000000, 0.0000000, 1.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0250000, 0.0000000, 0.0000000,
    0.0000000, 0.0000000, 0.0000000, 0.0000000, 1.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0250000, 0.0000000,
    0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 1.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0250000,
    0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.9810000, 0.0000000, 1.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0122625, 0.0000000,
    0.0000000, 0.0000000, 0.0000000, -0.9810000, 0.0000000, 0.0000000, 0.0000000, 1.0000000, 0.0000000, -0.0122625, 0.0000000, 0.0000000,
    0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 1.0000000, 0.0000000, 0.0000000, 0.0000000,
    0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 1.0000000, 0.0000000, 0.0000000,
    0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 1.0000000, 0.0000000,
    0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 1.0000000
];

#[rustfmt::skip]
const B : [f64;48] = [
    -0.0007069, 0.0007773, 0.0007091, -0.0007795,
    0.0007034, 0.0007747, -0.0007042, -0.0007739,
    0.0052554, 0.0052554, 0.0052554, 0.0052554,
    -0.1720966, -0.1895213, 0.1722891, 0.1893288,
    -0.1729419, 0.1901740, 0.1734809, -0.1907131,
    0.0123423, -0.0045148, -0.0174024, 0.0095748,
    -0.0565520, 0.0621869, 0.0567283, -0.0623632,
    0.0562756, 0.0619735, -0.0563386, -0.0619105,
    0.2102143, 0.2102143, 0.2102143, 0.2102143,
    -13.7677303, -15.1617018, 13.7831318, 15.1463003,
    -13.8353509, 15.2139209, 13.8784751, -15.2570451,
    0.9873856, -0.3611820, -1.3921880, 0.7659845
];

#[rustfmt::skip]
const Q: [f64; 12] = [ 100.0000000, 100.0000000, 100.0000000, 4.0000000, 4.0000000, 400.0000000, 4.0000000, 4.0000000, 4.0000000, 2.0408163, 2.0408163, 4.0000000, ];

#[rustfmt::skip]
const R: [f64; 4] = [4.0000000, 4.0000000, 4.0000000, 4.0000000];

const RHO: f64 = 5.0;
