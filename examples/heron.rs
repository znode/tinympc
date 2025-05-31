use log::debug;
use nalgebra::{DMatrix, DVector, vector};
use tinympc::TinySolver;

const NU: usize = 2;
const NX: usize = 3;
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
        DMatrix::from_element(NX, NH, -100.0),
        DMatrix::from_element(NX, NH, 100.0),
        DMatrix::from_element(NU, NH - 1, -10.0),
        DMatrix::from_element(NU, NH - 1, 10.0),
    );

    // Configure settings
    mpc.settings.max_iter = 100;

    // Constant reference through entire horizon
    let reference = vector![10.0, 10.0, 0.0];
    let mut xref = DMatrix::<f64>::zeros(NX, NH);

    for i in 0..NH {
        xref.set_column(i, &reference);
    }

    mpc.set_x_ref(xref.clone());

    // Dynamic state vector Nx
    let mut x = DVector::from_vec(vec![1.0, 1.0, 1.0]);

    let mut total_iters = 0;
    for k in 0..5000 {
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
        debug!("{u}");
        x = &a * x + &b * &u;

        total_iters += mpc.num_iters();
    }

    println!("Total iterations: {total_iters}");
}

#[rustfmt::skip]
const A:[f64; 9] = [
     1.0, 0.0, 0.0,
     0.0, 1.0, 0.5,
     0.0, 0.0, 1.0,
];

//20Hz
#[rustfmt::skip]
const B : [f64;6] = [
    0.025, 0.025,
    -0.01785714, 0.01785714,
    -0.07142857, 0.07142857
];

#[rustfmt::skip]
const Q: [f64; NX] = [ 100.0, 100.0, 20.0];

#[rustfmt::skip]
const R: [f64; NU] = [1.0, 1.0];

const RHO: f64 = 5.0;
