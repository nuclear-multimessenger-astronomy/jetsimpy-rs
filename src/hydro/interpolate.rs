use crate::constants::PI;
use crate::hydro::tools::Tool;

/// Bilinear interpolation of PDE solution data.
pub struct Interpolator {
    ntheta: usize,
    tmin: f64,
    tmax: f64,
    theta_min: f64,
    theta_max: f64,
}

impl Interpolator {
    pub fn new(theta: &[f64], ts: &[f64]) -> Self {
        Interpolator {
            ntheta: theta.len(),
            tmin: ts[0],
            tmax: *ts.last().unwrap(),
            theta_min: theta[0],
            theta_max: *theta.last().unwrap(),
        }
    }

    fn find_theta_index(&self, theta: f64, theta_data: &[f64], tool: &Tool) -> (usize, usize) {
        if theta < 0.0 || theta > PI {
            panic!("Interpolation: theta outside bounds.");
        } else if theta < self.theta_min {
            (0, 0)
        } else if theta > self.theta_max {
            (self.ntheta - 1, self.ntheta - 1)
        } else {
            tool.find_index(theta_data, theta)
        }
    }

    fn find_time_index(&self, t: f64, t_data: &[f64], tool: &Tool) -> (usize, usize) {
        if t < 0.0 || t > self.tmax {
            panic!("Interpolation: t outside bounds.");
        } else if t < self.tmin {
            (0, 0)
        } else {
            tool.find_index(t_data, t)
        }
    }

    pub fn interpolate_y(
        &self,
        t: f64,
        theta: f64,
        y_index: usize,
        y_data: &[Vec<Vec<f64>>],
        t_data: &[f64],
        theta_data: &[f64],
        tool: &Tool,
    ) -> f64 {
        let (theta_idx1, theta_idx2) = self.find_theta_index(theta, theta_data, tool);
        let (t_idx1, t_idx2) = self.find_time_index(t, t_data, tool);

        let t1 = t_data[t_idx1];
        let t2 = t_data[t_idx2];
        let th1 = theta_data[theta_idx1];
        let th2 = theta_data[theta_idx2];

        let y11 = y_data[y_index][theta_idx1][t_idx1];
        let y12 = y_data[y_index][theta_idx2][t_idx1];
        let y21 = y_data[y_index][theta_idx1][t_idx2];
        let y22 = y_data[y_index][theta_idx2][t_idx2];

        // Interpolate over theta
        let y1 = if theta_idx1 == theta_idx2 {
            y11
        } else {
            tool.linear(theta, th1, th2, y11, y12)
        };
        let y2 = if theta_idx1 == theta_idx2 {
            y21
        } else {
            tool.linear(theta, th1, th2, y21, y22)
        };

        // Interpolate over t
        if t_idx1 == t_idx2 {
            y1
        } else {
            tool.linear(t, t1, t2, y1, y2)
        }
    }
}
