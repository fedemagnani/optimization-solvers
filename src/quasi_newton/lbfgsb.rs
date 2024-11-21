use super::*;

pub struct Lbfgsb {
    n: i32,
    m: i32,
    l: Vec<Floating>,
    u: Vec<Floating>,
    nbd: Vec<i32>,
    factr: Floating,
    pgtol: Floating,
    wa: Vec<Floating>,
    iwa: Vec<i32>,
    task: Vec<i8>,
    iprint: i32,
    csave: Vec<i8>,
    lsave: Vec<i32>,
    isave: Vec<i32>,
    dsave: Vec<Floating>,
    max_iter: u32,
}

impl Lbfgsb {
    pub fn minimize(
        &mut self,
        oracle: impl Fn(&DVector<Floating>) -> FuncEvalMultivariate,
        x0: &mut DVector<Floating>,
    ) -> Result<(), SolverError> {
        let eval = oracle(x0);
        let mut f = *eval.f();
        let mut g = eval.g().clone_owned();
        stringfy(&mut self.task);
        let k = 1.0e-10;
        // start of the loop
        loop {
            // callign the fortran routine
            unsafe {
                ffi::setulb_(
                    &self.n,
                    &self.m,
                    x0.as_mut_ptr(),
                    self.l.as_ptr(),
                    self.u.as_ptr(),
                    self.nbd.as_ptr(),
                    &f,
                    g.as_ptr(),
                    &self.factr,
                    &self.pgtol,
                    self.wa.as_mut_ptr(),
                    self.iwa.as_mut_ptr(),
                    self.task.as_mut_ptr(),
                    &self.iprint,
                    self.csave.as_mut_ptr(),
                    self.lsave.as_mut_ptr(),
                    self.isave.as_mut_ptr(),
                    self.dsave.as_mut_ptr(),
                )
            }

            // converting to rust string
            let tsk = unsafe { CStr::from_ptr(self.task.as_ptr()).to_string_lossy() };
            // println!("{}", tsk);
            if &tsk[0..2] == "FG" {
                let eval = oracle(x0);
                f = *eval.f();
                g.copy_from_slice(eval.g().as_slice());
            }
            if &tsk[0..5] == "NEW_X" && self.max_iter == 0 && self.dsave[11] <= k * f
            //?
            {
                // println!("THE PROJECTED GRADIENT IS SUFFICIENTLY SMALL");
                return Ok(());
            }
            if self.max_iter > 0 && self.isave[29] >= self.max_iter as i32 {
                return Err(SolverError::MaxIterReached);
            }
            if &tsk[0..4] == "CONV" {
                return Ok(());
            }
            if &tsk[0..5] == "ERROR" {
                return Err(SolverError::ErrorInputParams);
            }
            if &tsk[0..8] == "ABNORMAL" {
                return Err(SolverError::AbnormalTermination);
            }
        }
    }
    // constructor requres three mendatory parameter which is the initial solution, function and the gradient function
    pub fn new(n: usize) -> Self {
        // creating lbfgs struct
        Lbfgsb {
            m: 5,
            l: vec![0.0; n],
            u: vec![0.0; n],
            nbd: vec![0; n],
            factr: 0.0,
            pgtol: 0.0,
            wa: vec![0.0; 2 * 17 * n + 5 * n + 11 * 17 * 17 + 8 * 17],
            iwa: vec![0; 3 * n],
            task: vec![0; 60],
            iprint: -1,
            csave: vec![0; 60],
            lsave: vec![0, 0, 0, 0],
            isave: vec![0; 44],
            dsave: vec![0.0; 29],
            max_iter: 0,
            n: n as i32,
        }
    }

    // this function will start the optimization algorithm

    // this function is used to set lower bounds to a variable
    pub fn set_lower_bound(&mut self, index: usize, value: Floating) {
        if self.nbd[index] == 1 || self.nbd[index] == 2 {
            // println!("variable already has Lower Bound");
        } else {
            let temp = self.nbd[index] - 1;
            self.nbd[index] = if temp < 0 { -temp } else { temp };
            self.l[index] = value;
        }
    }
    // this function is used to set upper bounds to a variable
    pub fn set_upper_bound(&mut self, index: usize, value: Floating) {
        if self.nbd[index] == 3 || self.nbd[index] == 2 {
            // println!("variable already has Lower Bound");
        } else {
            self.nbd[index] = 3 - self.nbd[index];
            self.u[index] = value;
        }
    }
    // set the verbosity level
    pub fn set_verbosity(&mut self, l: i32) {
        self.iprint = l;
    }
    // set termination tolerance
    // 1.0e12 for low accuracy
    // 1.0e7  for moderate accuracy
    // 1.0e1  for extremely high accuracy
    pub fn set_factr(&mut self, t: Floating) {
        self.factr = t;
    }
    // set tolerance of projection gradient
    pub fn set_pgtol(&mut self, t: Floating) {
        self.pgtol = t;
    }
    // set max iteration
    pub fn max_iteration(&mut self, i: u32) {
        self.max_iter = i;
    }
    // set maximum number of variable metric corrections
    // The range  3 <= m <= 20 is recommended
    pub fn set_m(&mut self, m: i32) {
        self.m = m;
    }
}

#[inline]
pub fn stringfy(task: &mut [i8]) {
    unsafe {
        ffi_string::stringfy_(task.as_mut_ptr());
    }
}
