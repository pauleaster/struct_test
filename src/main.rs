use colour::unnamed::Colour;
use itertools::{Itertools, izip};

// use core::num::dec2flt::float;
use std::time::Instant;

use rand::distributions::{Distribution, Uniform};


use std::f64::consts::PI;
use std::env;
use std::process::exit as exit;

use std::collections::{HashMap, HashSet};
use std::iter::FromIterator;


// use geo::{LineString, Polygon, prelude::Contains};

// extern crate colorful;


// use colorful::Colorful;

const MAX_NUM : f64 = 1e6;
const MIN_NUM : f64 = 0.0;

// const GEO_SCALE: f64 = 10000.0;

// static mut GLOB_CLONE_COORDINATE: f64 = 0.0;
// static mut GLOB_CLONE_COORDINATE_VECTOR: f64 = 0.0;
// static mut GLOB_CALC_MAG_COORDINATE: f64 = 0.0;
// static mut COORDINATE_NEW_COUNT: usize = 0;
// static mut COORDINATE_VECTOR_NEW_COUNT: usize = 0;
// static mut GLOB_NEW_COORDINATE: f64 = 0.0;
// static mut GLOB_NEW_COORDINATE_VECTOR: f64 = 0.0;
static mut GLOB_NEW_COORDINATE_DIFFERENCES: f64 = 0.0;
// static mut GLOB_REPOSITION_COORDINATE_VECTOR: f64 = 0.0;
static mut GLOB_NEW_COORDINATE_DIFFERENCES_LOOP_ONLY: f64 = 0.0;
static mut GLOB_NEW_COORDINATE_DIFFERENCES_MIN_MAX_ONLY: f64 = 0.0;
// static mut GLOB_NEW_COORDINATE_DIFFERENCES_EDGE_DOTS_ONLY: f64 = 0.0;


fn float_equals(x: f64, y: f64, eps: f64) -> bool{
    
    (x-y).abs() < eps.abs()
}

fn vec_copy_usize( data: &[usize]) -> Vec<usize> {

    data.iter().copied().collect()
}

fn vec_usize_print( data: &[usize]) {

    let size = data.len();

    colour::dark_red!("( ");
    for val in data.iter().take(size - 1) {
        colour::dark_red!("{}, ",val);
    }
    if size > 0 {
        colour::dark_red!("{} ",&data[size-1]);
    }
    colour::dark_red!(")");
}

fn vec_2d_copy_usize( data: &[Vec<usize>]) -> Vec<Vec<usize>> {

    let mut result: Vec<Vec<usize>> = Vec::new();


    for uvec in data.iter().map(|x| vec_copy_usize(x)){
        result.push(uvec);
    }

    result

}

fn vec_copy_f64( data: &[f64]) -> Vec<f64> {

    data.iter().copied().collect()
}

fn uniquely_sorted( vector: &[usize]) -> Vec<usize>{
    HashSet::<&usize>::from_iter(vector)
    .into_iter()
    .sorted()
    .cloned()
    .collect()
}

struct SphericalAngles {
    theta : f64,
    phi : f64,
    num_points: usize,
    index: i32,
}

impl SphericalAngles {

    fn copy(&self) -> SphericalAngles {
        SphericalAngles {
            theta: self.theta,
            phi : self.phi,
            num_points: self.num_points,
            index: self.index,
        } 
    }


}


struct GoldenSpiral {

    curr: SphericalAngles,

}

impl GoldenSpiral {

    fn new(num_points: usize) -> GoldenSpiral {
        GoldenSpiral {
            curr : SphericalAngles {
                theta: 0.0,
                phi : 0.0,
                num_points,
                index: -1,
            },

        }
    }
}

impl Iterator for GoldenSpiral {

    type Item = SphericalAngles;

    fn next( &mut self) -> Option<Self::Item> {

        if self.curr.index < (self.curr.num_points - 1) as i32{
            self.curr.index += 1;
            self.curr.theta = PI * (1.0 + 5.0_f64.sqrt() * self.curr.index as f64 );
            self.curr.phi = (1.0 - 2.0 * self.curr.index as f64 / self.curr.num_points as f64).acos();
            return Some(self.curr.copy());
        }
        None

        
    }


}

pub type NonAndParallelVertices = (bool, usize, usize, Option<Vec<(usize,usize)>>, Option<Vec<(usize,usize)>>);


#[derive(Debug)]
struct Coordinate {

    x: f64,
    y: f64,
    z: f64,
    mag: f64,
}



impl Coordinate {


    fn new(x : f64, y: f64, z: f64 ) -> Coordinate{

        Coordinate {
            x, 
            y,
            z,
            mag : Coordinate::calc_mag(x,y,z),
        }
    }

    fn new_from_vector( coordinate : Vec<f64>) -> Coordinate {

        // Only take the first three components if more are given
        let mut size = coordinate.len();
        if size > 3 {
            size = 3;
        }
        let mut x:f64 = 0.0;
        let mut y:f64 = 0.0;
        let mut z:f64 = 0.0;

        for (idx, &value) in  coordinate.iter().enumerate().take(size) {
            if idx == 0 {
                x = value;
            }
            if idx == 1 {
                y = value;
            }
            if idx == 2 {
                z = value;
            }
        }
        Coordinate::new(x,y,z)
    }


    fn unit(x : f64, y: f64, z: f64 ) -> Coordinate{

        let initial_mag = Coordinate::calc_mag(x,y,z);
        Coordinate {
            x: x/initial_mag, 
            y: y/initial_mag,
            z: z/initial_mag,
            mag : 1.0,
        }
    }

    fn zero() -> Coordinate{ Coordinate::new(0.,0.,0.) }

    fn new_random() -> Coordinate {

        let mut rng = rand::thread_rng();
        // let mut rng = StdRng::seed_from_u64(1);
        let theta_range = Uniform::from(0.0..2.0 * PI);
        let phi_range = Uniform::from(0.0..PI);

        let theta = theta_range.sample(&mut rng);
        let phi = phi_range.sample(&mut rng);

        // println!("theta = {}, phi = {}",theta,phi);

        Coordinate::new_from_spherical_coordinates(1.0,  theta, phi)
        
    }

    fn new_from_spherical_coordinates( r: f64, theta: f64, phi: f64) -> Coordinate{


        let x : f64 = r * theta.cos() * phi.sin();
        let y : f64 = r * theta.sin() * phi.sin();
        let z : f64 = r * phi.cos();

        let mag: f64 = r;

        Coordinate {
                x,
                y,
                z,
                mag,
            }

    }

    fn project_onto_xz(&self) -> Coordinate {
        
        let x = self.x;
        let y = 0.;
        let z = self.z;
        let mag = Coordinate::calc_mag(x,y,z);

        
        Coordinate {
            x,
            y,
            z,
            mag,
        }
    }

    fn project_onto_xy(&self) -> Coordinate {
        
        let x = self.x;
        let y = self.y;
        let z = 0.0;
        let mag = Coordinate::calc_mag(x,y,z);

        
        Coordinate {
            x,
            y,
            z,
            mag,
        }
    }

    fn clone(&self) -> Coordinate {
        
        let x = self.x;
        let y = self.y;
        let z = self.z;
        let mag = self.mag;

        Coordinate {
            x,
            y,
            z,
            mag,
        }
    }

    fn copy(&self) -> Coordinate {
        self.clone()
    }

    fn add(&self, pt:&Coordinate) -> Coordinate {

        let x:f64 = self.x + pt.x;
        let y:f64 = self.y + pt.y;
        let z:f64 = self.z + pt.z;


        Coordinate {
            x,
            y,
            z,
            mag : Coordinate::calc_mag(x,y,z),
        }
        
    }

    fn sub(&self, pt:&Coordinate) -> Coordinate {

        let x:f64 = self.x - pt.x;
        let y:f64 = self.y - pt.y;
        let z:f64 = self.z - pt.z;

        Coordinate {
            x,
            y,
            z,
            mag : Coordinate::calc_mag(x,y,z),
        }
        
    }


    fn calc_mag(x:f64,y:f64,z:f64) -> f64 {
        
        (x.powi(2)  + y.powi(2) + z.powi(2)).sqrt()


    }

    fn mult(&self, scale: f64) -> Coordinate {

        let x:f64 = scale * self.x;
        let y:f64 = scale * self.y;
        let z:f64 = scale * self.z;

        Coordinate {
            x,
            y,
            z,
            mag : Coordinate::calc_mag(x,y,z),
        }

    }

    

    fn make_unit_vector(&self) -> Coordinate{

        // mag = self.mag;

        self.mult(1.0 / self.mag)
    }

    fn first_quadrant(self) -> Coordinate {

        Coordinate {
            x: self.x.abs() ,
            y: self.y.abs() ,
            z: self.z.abs(), 
            mag: self.mag,
        }
        
    }

    fn equal(&self, pt: &Coordinate) -> bool {
        
        let eps = 1e-6_f64;

        self.sub(pt).mag < eps

    }

    fn equal_or_inverted( &self, pt: &Coordinate) -> bool {     
        self.equal(pt) | self.equal(&pt.mult(-1.0))
    }

    fn print(&self, ch:char, precision: usize) {

        let field:usize = precision + 4; 
    
        colour::blue!("( {:field$.precision$} , {:field$.precision$} , {:field$.precision$} )",
                    self.x,self.y,self.z, precision=precision,field=field);
        colour::dark_magenta!(", |x| = {:field$.precision$}{}",
                    self.mag,ch,precision=precision,field=field)
        
    }

    fn sub_float(&self, value:f64) -> Coordinate {

        let x = self.x - value ;
        let y = self.y - value ;
        let z = self.z - value ;
        
        
        Coordinate {
            x,
            y,
            z, 
            mag: Coordinate::calc_mag(x,y,z),
        }
    }

    fn dot(&self, pt:&Coordinate) -> f64 {

        self.x * pt.x + self.y * pt.y + self.z * pt.z
    }

    fn unit_dot(&self, pt: &Coordinate) -> f64{
        self.dot(pt) / self.mag / pt.mag
    }

    fn unit_cross(&self, pt: &Coordinate) -> Coordinate{

        let x = self.y * pt.z - self.z * pt.y;
        let y = self.z * pt.x - self.x * pt.z;
        let z = self.x * pt.y - self.y * pt.x;
        let mag = Coordinate::calc_mag(x,y,z);
        
        Coordinate{
            x: x / mag,
            y: y / mag,
            z: z / mag,
            mag: 1.0,
        }
    }

    fn expand(self: &Coordinate) -> Vec<f64> {
        vec!(self.x, self.y, self.z)
    }

    fn angle(&self, pt:&Coordinate) -> f64 {

        self.unit_dot(pt).acos() * 180.0 / PI 
    }

    fn get_by_index( self: &Coordinate, index: usize) -> f64 {

        match index {
            0 => self.x,
            1 => self.y,
            2 => self.z,
            3 => self.mag,
            _ => panic!("Coordinate::get_by_index must have an index <=3, the value of index is {}",index),
        }
    }

    fn calculate_2d_determinant(a: &Coordinate, b: &Coordinate) -> f64 {
        // This function calculates the 2x2 determinant of vector a and b by
        // ignoring the z components.
        // If the z components are non-zero then the output is meaningless.
        if ! float_equals(a.z, 0.0, 1e-6) | ! float_equals(b.z, 0.0, 1e-6) {
            panic!("2D determinant called with non zero z values, a.z = {:0.8}, b.z = {:0.8}", a.z,b.z);
        }
        else {
            a.x * b.y - a.y * b.x
        }
    }

    fn calculate_determinant(a:&Coordinate, b:&Coordinate, c:&Coordinate) -> f64 {
        
        let mut a2d:Coordinate = a.clone();
        let mut b2d:Coordinate = b.clone();
        let mut c2d:Coordinate = c.clone();

        a2d.z = 0.0;
        b2d.z = 0.0;
        c2d.z = 0.0;

        a.z * Coordinate::calculate_2d_determinant(&b2d, &c2d) 
            - b.z * Coordinate::calculate_2d_determinant(&a2d, &c2d) 
            + c.z * Coordinate::calculate_2d_determinant(&a2d, &b2d)

    }

    // fn project_to_xy_convert_to_geo_coordinate(& self, geo_scale: f64) -> geo::Coordinate<f64> {
    //     // println!("x,y={},{}",self.x  * geo_scale, self.y  * geo_scale);
    //     geo::Coordinate {
    //         x: self.x  * geo_scale,
    //         y: self.y * geo_scale,
            
    //     }
    // }

    fn is_parallel_to(&self, coordinate: & Coordinate) -> bool {

        float_equals(self.dot(coordinate), self.mag * coordinate.mag, 1e-10)

    }

}


#[derive(Debug)]
struct CoordinateVector {

    size: usize,
    data:  Vec<Coordinate>,
}

impl CoordinateVector {
    
    
    fn new(data : Vec<Coordinate> ) -> CoordinateVector{

        CoordinateVector{

            size: data.len(),
            data, // note this should move data
        }
    }




    fn new_from_empty() -> CoordinateVector {
        
        let data: Vec<Coordinate> = Vec::new();
        let size:usize = 0;

        CoordinateVector {
            size,
            data,
        }
    }

    fn new_from_random_vertices(vertices: usize) -> CoordinateVector {

        if vertices < 2 {
            panic!("Must have more than one vertex")
        } else {

            let mut data : Vec<Coordinate> = vec![Coordinate::new(0.0, 0.0, 1.0), Coordinate::new_random().project_onto_xz().make_unit_vector()]; 
            for _ in 2..vertices{
                data.push(Coordinate::new_random());
            }
            if data.len() != vertices{
                panic!("The length of the CoordinateVector is wrong, it should be {} but it is {}.",vertices,data.len());
            }
            CoordinateVector::new(data)

        }

    }

    fn new_from_fixed_sequence(vertices: usize) -> CoordinateVector {

        if vertices < 3 {
            panic!("Must have more than two vertices")
        } else {

            let mut data : Vec<Coordinate> = Vec::new(); 
            let m = ((vertices - 2) as f64).sqrt() as usize;
            let dtheta = 2.0 * PI / ((m as f64) + 2.0 );
            let dphi = PI / ((m as f64) + 2.0 );
            // First the 0th vertex at z=1
            data.push(Coordinate::new_from_spherical_coordinates(1.0, 0.0 ,0.0 ));
            println!("({},{}): theta = {:0.6}, phi = {:0.6}", 0, 0, 0.0, 0.0);
            let cx = &data[data.len()-1].x;
            let cy = &data[data.len()-1].y;
            let cz = &data[data.len()-1].z;
            println!("({:0.6}, {:0.6}, {:0.6} )", cx, cy, cz);
            // Second the 1st vertex at x=1
            data.push(Coordinate::new_from_spherical_coordinates(1.0, 0.0 , PI/2.0 ));
            println!("({},{}): theta = {:0.6}, phi = {:0.6}", 0, 1, 0.0, PI/2.0);
            let cx = &data[data.len()-1].x;
            let cy = &data[data.len()-1].y;
            let cz = &data[data.len()-1].z;
            println!("({:0.6}, {:0.6}, {:0.6} )", cx, cy, cz);
            for itheta in 1..m + 1 { 
                for iphi in 1..m + 1 {
                    let theta = (itheta as f64 )* dtheta;
                    let phi = (iphi as f64) * dphi;
                    println!("({},{}): theta = {:0.6}, phi = {:0.6}", itheta, iphi, theta, phi);
                    data.push(Coordinate::new_from_spherical_coordinates(1.0, theta, phi));
                    colour::yellow_ln!(" length = {}",data.len());
                    let cx = &data[data.len()-1].x;
                    let cy = &data[data.len()-1].y;
                    let cz = &data[data.len()-1].z;
                    println!("({:0.6}, {:0.6}, {:0.6} )", cx, cy, cz);

                }
            }
            if m*m  + 2 < vertices {
                for vertex in m*m + 2 ..vertices {
                    let itheta = (vertex - m*m - 2) as f64 + 0.5;
                    let iphi = (vertex - m*m - 2) as f64 + 0.5;
                    let theta = (itheta as f64 )* dtheta;
                    let phi = (iphi as f64) * dphi;
                    println!("theta = {:0.6}, phi = {:0.6}", theta, phi);
                    data.push(Coordinate::new_from_spherical_coordinates(1.0, theta, phi));
                    let cx = &data[data.len()-1].x;
                    let cy = &data[data.len()-1].y;
                    let cz = &data[data.len()-1].z;
                    println!("({:0.6}, {:0.6}, {:0.6} )", cx, cy, cz);
                }
            }
            if data.len() != vertices{
                panic!("The length of the CoordinateVector is wrong, it should be {} but it is {}.",vertices,data.len());
            }
            CoordinateVector::new(data)

        }

    }

    fn new_from_symmetric_fixed_sequence(num_vertices: usize) -> CoordinateVector {

        if num_vertices < 3 {
            panic!("Must have more than two vertices")
        } else {

            let mut data : Vec<Coordinate> = Vec::new(); 
            let mut cumulative_vertices : usize = 0;
            let r = 1.0;
            let theta = 0.0;
            let phi = 0.0;
            data.push(Coordinate::new_from_spherical_coordinates(r, theta, phi));
            colour::cyan_ln!("theta={:0.3}, phi={:0.3}", theta,phi);
            cumulative_vertices += 1;
            if num_vertices >= 8 {
                let num_pi_on_4 = (num_vertices - 2) / 3;
                if num_pi_on_4 > 0 {
                    let dtheta = 2.0 * PI / (num_pi_on_4 as f64);
                    let phi = PI / 4.0;
                    for i in 0..num_pi_on_4 {
                        let theta = (i as f64) * dtheta;
                        data.push(Coordinate::new_from_spherical_coordinates(r, theta, phi));
                        colour::cyan_ln!("theta={:0.3}, phi={:0.3}", theta,phi);
                    }
                    let phi = 3.0 * PI / 4.0;
                    for i in 0..num_pi_on_4 {
                        let theta = (i as f64) * dtheta;
                        if num_vertices == 5 {
                            let theta = PI;
                            data.push(Coordinate::new_from_spherical_coordinates(r, theta, phi));
                            colour::cyan_ln!("theta={:0.3}, phi={:0.3}", theta,phi);
                        } else {
                            data.push(Coordinate::new_from_spherical_coordinates(r, theta, phi));
                            colour::cyan_ln!("theta={:0.3}, phi={:0.3}", theta,phi);
                        }
                        
                    }
                    cumulative_vertices += 2 * num_pi_on_4;
                }
            }
            
            let phi = PI / 2.0;
            let num_pi_on_2 = num_vertices - cumulative_vertices - 1;
            let mut dtheta = 2.0 * PI / (num_pi_on_2 as f64);
            if num_vertices == 4 {
                dtheta = 2.0 * PI / ((num_pi_on_2 + 1) as f64);
            }
            // colour::magenta_ln!("num_vertices = {}, num_pi_on_4 = {}, cumulative_vertices = {}", num_vertices, num_pi_on_4, cumulative_vertices);
            for i in 0..num_pi_on_2 {
                let theta = (i as f64) * dtheta;
                data.push(Coordinate::new_from_spherical_coordinates(r, theta, phi));
                colour::cyan_ln!("theta={:0.3}, phi={:0.3}", theta,phi);
            }
            cumulative_vertices += num_pi_on_2;
            let phi = PI;
            let theta = 0.0;
            data.push(Coordinate::new_from_spherical_coordinates(r, theta, phi));
            colour::cyan_ln!("theta={:0.3}, phi={:0.3}", theta,phi);
            cumulative_vertices += 1;
            if (cumulative_vertices != num_vertices) | (data.len() != num_vertices) {
                panic!("(cumulative_vertices != num_vertices) | (data.len() != num_vertices), ({} != {}) | ({} != {})",cumulative_vertices,num_vertices,data.len(),num_vertices);
            }
            CoordinateVector::new(data)
        }
    }

    fn new_from_golden_spiral(num_vertices: usize) -> CoordinateVector {

        if num_vertices < 3 {
            panic!("Must have more than two vertices")
        } else {
            let mut cv : CoordinateVector = CoordinateVector::new_from_empty(); 

            for sa in GoldenSpiral::new(num_vertices) {
                let mut c = Coordinate::new_from_spherical_coordinates(1.0,sa.theta,sa.phi);
                if sa.index == 1 {
                    c = c.project_onto_xz().make_unit_vector();
                }
                cv.push(&c);

            }
            cv
        }

    }



    fn clone(&self) -> CoordinateVector {

        let mut data:  Vec<Coordinate> = Vec::new();

        for coordinate in &self.data {
            data.push(coordinate.clone())
        }
        CoordinateVector::new(data)
    }

    fn copy(&self) -> CoordinateVector {
        self.clone()
    }

    
    fn zero(&self) -> CoordinateVector{

        let mut zero_vectors = self.clone();

        for index in 0..self.size {
            zero_vectors.data[index] = Coordinate::zero();
        }
        zero_vectors
    }

    fn push( & mut self, pt : & Coordinate) {

        self.data.push(pt.clone());
        self.size += 1;

    }

    fn sum(& self) -> Coordinate {
        
        let mut sum = Coordinate::zero();
        for coordinate in &self.data {
            sum = sum.add(coordinate);
        }
        sum
    }

    fn print(&self, precision: usize) {

        colour::yellow_ln!("CoordinateVector::print()");
        colour::yellow_ln!("Length = {:3}",self.size);
        for (idx, coordinate) in self.data.iter().enumerate() {
            colour::dark_red!("{}, coord:",idx);
            coordinate.print('\n', precision);
        }
        colour::yellow_ln!("Sum = ");
        self.sum().print('\n', precision);
    }

    fn max_mag(&self) -> f64 {
        self.clone().data.iter().map(|x| x.mag).fold(-1e6,f64::max) // https://stackoverflow.com/a/66455028/1542485
        
    }

    fn reposition (&self, differences: &CoordinateDifferences, scale:f64) -> (CoordinateVector, f64) {

        let result = self; //.clone();
        let mut dx = self.zero();
        let mut dx_parallel = self.zero();
        let mut true_dx = self.zero();
        let mut new_result = self.zero();
        for (idx1, idx2, delta_vector) in izip!(&differences.first_index,&differences.second_index,&differences.data){
            if *idx1 == 0 {
                dx.data[*idx1]= Coordinate::zero();
            } else if *idx1 == 1 {
                dx.data[*idx1]  = dx.data[*idx1].add(&delta_vector.mult(delta_vector.mag.powi(-3) * scale)).project_onto_xz();
                if dx.data[*idx1].y.abs() > 1e-3 {
                    println!("The y component of the 1st vector is not zero, the vector is :");
                    dx.data[*idx1].print(' ', 3);
                    panic!();
                }
            } else {
                dx.data[*idx1]  = dx.data[*idx1].add(&delta_vector.mult(delta_vector.mag.powi(-3) * scale)); // du_hat/|u|^2
            }
            
            if *idx2 == 0 {
                dx.data[*idx2]= Coordinate::zero();
            } else if *idx2 == 1 {
                dx.data[*idx2]  = dx.data[*idx2].add(&delta_vector.mult(-1.0 * delta_vector.mag.powi(-3) * scale)).project_onto_xz();
                if dx.data[*idx2].y.abs() > 1e-3 {
                    println!("The y component of the 1st dx vector is not zero, dx = ");
                    dx.data[*idx2].print('\n', 3);
                    panic!();
                }
            } else {
                dx.data[*idx2]  = dx.data[*idx2].add(&delta_vector.mult(-1.0 * delta_vector.mag.powi(-3) * scale)); // du_hat/|u|^2
            }
        }
        for (idx, dx_val) in dx.data.iter().enumerate() {
            dx_parallel.data[idx] = dx_val.sub(&result.data[idx].mult(dx_val.dot(&result.data[idx])));
            if idx == 1 {
                if result.data[idx].y.abs() > 1e-3 {
                    println!("1st vector is not on xz plane, v = ");
                    result.data[idx].print('\n', 3);
                    panic!();
                }
                if dx_parallel.data[idx].y.abs() > 1e-3 {
                    println!("dx parallel of the 1st vector is not on xz plane, dx_para = ");
                    dx_parallel.data[idx].print('\n', 3);
                    panic!();
                }
                if result.data[idx].add(&dx_parallel.data[idx]).make_unit_vector().y.abs() > 1e-3 {
                    println!("adjusted 1st vector is not on xz plane, dx_para = ");
                    result.data[idx].add(&dx_parallel.data[idx]).make_unit_vector().print('\n', 3);
                    panic!();
                }
            }
            new_result.data[idx] = result.data[idx].add(&dx_parallel.data[idx]).make_unit_vector();
            true_dx.data[idx] = new_result.data[idx].sub(&result.data[idx]);
        }
        (new_result, true_dx.max_mag())
    }

    fn indexed_coordinate(&self, index: usize) -> Coordinate{

        if index < self.size {
            self.data[index].copy()
        } else {
            panic!("Index out of bounds, requesting index {} and maximum index is {}",index,self.size);
        }
        
    }

    fn remove(& mut self, index: usize) {
        
        if index < self.size {
            self.data.remove(index);
            self.size -= 1;
        } else {
            panic!("Index out of bounds, requesting index {} and maximum index is {}",index,self.size);
        }

    }
    
    fn project_to_xy_convert_to_geo_2d( & self, geo_scale: f64) -> Vec<(f64, f64)> { 

        let mut poly_vec = self.data.iter().map(|c| (c.x,c.y)).collect::<Vec<(f64,f64)>>();
        println!("poly x0,y0={},{}",self.data[0].x * geo_scale,self.data[0].y * geo_scale);
        poly_vec.push((self.data[0].x * geo_scale,self.data[0].y * geo_scale));
        poly_vec
    }

    // fn project_to_xy_onto_geo_polygon( & self, geo_scale: f64) -> Polygon<f64> {


    //     Polygon::new(
    //         LineString::from(self.project_to_xy_convert_to_geo_2d(geo_scale)),
    //         vec![],
    //     )
    // }

    fn calc_centroid( & self) -> Coordinate {

        self.sum().mult(1.0 / self.size as f64)
    }

    

    fn get_non_and_parallel_vertices( & self) -> NonAndParallelVertices {

        if self.size <= 2 {
            panic!("Doesn't make sense to calculate whether two vertices or less are parallel! Size = {}",self.size);
        }
        let ref_coordinate = self.calc_centroid();
        let mut non_para_count: usize = 0;
        let mut para_count: usize = 0;
        let mut all_perpendicular = true;
        let mut non_parallel_indices: Vec<(usize, usize)> = Vec::new();
        let mut parallel_indices: Vec<(usize, usize)> = Vec::new();
        for i in 0..self.size - 1 {
            for j in i + 1..self.size {
                let vector1 = self.indexed_coordinate(i).sub(&ref_coordinate);
                let vector2 = self.indexed_coordinate(i).sub(&ref_coordinate);
                let is_para = float_equals(vector1.dot(&vector2), 0.0, 1e-10);
                if is_para {
                    para_count += 1;
                    parallel_indices.push((i,j));
                    all_perpendicular = false;
                } else {
                    non_para_count += 1;
                    non_parallel_indices.push((i,j));
                }
            }
        }
        if para_count + non_para_count != self.size {
            panic!("Parallel and non parallel count does not add up to self.size, para_count {} + non_para_count {} != self.size {}",para_count,non_para_count,self.size);
        }
        let opt_parallel_indices = match para_count {
            0 => None,
            _ => Some(parallel_indices),
        };
        let opt_non_parallel_indices = match non_para_count {
            0 => None,
            _ => Some(non_parallel_indices),
        };
        (all_perpendicular, non_para_count, para_count, opt_non_parallel_indices, opt_parallel_indices)
    }
    
    
    fn co_planar( & self) -> bool {

        match self.size as i32 - 3 {
            -3..=-1 => false,
            0 => true,
            1.. => {
                let (all_perp, non_para_count, para_count, opt_non_parallel_indices, opt_parallel_indices) = self.get_non_and_parallel_vertices();
                // make sure that the basis vectors are not parallel with each other!
                let mut result = true;
                if non_para_count > 0 {
                    let ref_coordinate = self.calc_centroid();
                    let (basis_index_1, basis_index_2) = match opt_non_parallel_indices {
                        Some(v) => v[0],
                        None => panic!("non_para_count is larger than zero, but opt_non_parallel_indices is None!, non_para_count = {}", non_para_count),
                    };
                    let basis_1 = self.indexed_coordinate(basis_index_1).sub(&ref_coordinate).make_unit_vector();
                    let basis_2 = self.indexed_coordinate(basis_index_2).sub(&ref_coordinate).make_unit_vector();
                    let normal = basis_1.unit_cross(&basis_2);
                    for index in 3..self.size {
                        if float_equals(self.indexed_coordinate(index).sub(&ref_coordinate).dot(&normal),0.0,1e-10) {
                            continue;
                        } else {
                            result =  false;
                            break;
                        }
                    }
                } else { // non_para_count == 0!
                    panic!("non_para_count == 0 which does not make sense!")
                }
                result
            },
            _ => panic!("self.size - 3 < -3"),
        }
    }


    fn vertices_in_xy_plane(&self, opt_coordinate: & Option <& Coordinate>) -> bool {

        let coordinate_in_xy = match opt_coordinate {
            Some(coordinate) => float_equals(coordinate.z, 0.0, 1e-10),
            None => true,
        };
        if !coordinate_in_xy {
            return false
        }
        for i in 0..self.size {
            if !float_equals(self.indexed_coordinate(i).z, 0.0, 1e-10) {
                return false;
            }
        }
        true
    }

    fn min_max_xy( & self) -> ((f64, f64), (f64, f64)) {

        
        let mut min_x = self.indexed_coordinate(0).x;
        let mut min_y = self.indexed_coordinate(0).y;
        let mut max_x = self.indexed_coordinate(0).x;
        let mut max_y = self.indexed_coordinate(0).y;

        if self.size > 1 {
            for i in 1.. self.size {
                let coord = self.indexed_coordinate(i);
                if coord.x > max_x {
                    max_x = coord.x;
                }
                if coord.x < min_x {
                    min_x = coord.x;
                }
                if coord.y > max_y {
                    max_y = coord.y;
                }
                if coord.y < min_y {
                    min_y = coord.y;
                }
            }
        }
        ((min_x,min_y),(max_x,max_y))
    }

    
    fn sort_polygon_xy( & self ) -> CoordinateVector {

        let poly_ref = self.calc_centroid();
        let mut angles: Vec<f64> = Vec::new();
        let indices:Vec<f64> = (0..self.size).map(|x| x as f64).collect();

        let mut sorted_poly = CoordinateVector::new_from_empty();

        for i in 0..self.size {
            let coord = self.indexed_coordinate(i).sub(&poly_ref);
            angles.push(coord.y.atan2(coord.x));
        }

        let mut angle_indices:Vec<(&f64,&f64)> = angles.iter().zip(indices.iter()).collect();
        angle_indices.sort_by(|a,b| a.0.partial_cmp(b.0).unwrap());
        for ai in angle_indices.into_iter() {
            let i = *ai.1 as usize;
            sorted_poly.push(&self.indexed_coordinate(i));
        }
        sorted_poly
    }

    fn calc_polygon_beta(x: f64, y: f64, c1: &Coordinate, c2: &Coordinate) -> f64 {


        c2.x - x + (y - c2.y) * (c1.x - c2.x) / (c1.y - c2.y)
    }
    
    fn polygon_xy_contains(& self, coordinate: & Coordinate) -> bool {
        
        // this probably only works on convex polygons which is all that is expected here!
        if !self.vertices_in_xy_plane(&Some(coordinate)) {
            panic!("Unexpected data found, a vertex was not in the xy plane!");
        }
        let ((min_x, min_y), (max_x, max_y)) = self.min_max_xy();

        if (coordinate.y > max_y) | (coordinate.y < min_y) {
            return false;
        }
        if (coordinate.x > max_x) | (coordinate.x < min_x) {
            return false;
        }
        let x = coordinate.x;
        let y = coordinate.y;

        let mut sorted_polygon = self.sort_polygon_xy();
        let first_vertex = sorted_polygon.indexed_coordinate(0);
        let last_vertex = sorted_polygon.indexed_coordinate(sorted_polygon.size - 1);
        if !first_vertex.equal(&last_vertex) {
            sorted_polygon.push(&first_vertex);
        }

        let mut polygon_beta_values: Vec<f64>=Vec::new();
        for i in 0..sorted_polygon.size - 1 {
            let c1 = sorted_polygon.indexed_coordinate(i);
            let c2 = sorted_polygon.indexed_coordinate(i+1);
            let min_y = c1.y.min(c2.y);
            let max_y = c1.y.max(c2.y);
            if y >= min_y && y <= max_y {
                polygon_beta_values.push(CoordinateVector::calc_polygon_beta(x,y,&c1,&c2));
            }
        }
        if polygon_beta_values.len() != 2 {
            panic!("polygon_beta_values.len() != 2, polygon_beta_values.len()={}",polygon_beta_values.len());
        }
        polygon_beta_values[0].signum() != polygon_beta_values[1].signum()

    }

    fn aligned_with_coordinate( &self, coordinate : & Coordinate) -> bool {

        for i in 0..self.size {
            if self.indexed_coordinate(i).is_parallel_to(coordinate) {
                return true;
            }
        }
        false
    }
    
}





struct EdgeSwapResult {
    edge1: Vec<f64>,
    edge2: Vec<f64>,
    swapped: bool,
}


#[derive(Debug)]
struct CoordinateDifferences {

    size: usize,
    data:  Vec<Coordinate>,
    first_index : Vec <usize>,
    second_index : Vec <usize>,
    mean_magnitude : f64,
    magnitude_range : f64,
    signs: Vec <f64>,
    dots : Vec <f64>,
}


impl CoordinateDifferences{
    
    fn new(coordinates: &CoordinateVector) -> CoordinateDifferences {

        let now = Instant::now();
        let mut data : Vec<Coordinate> = Vec::new();
        let mut first_idx : Vec <usize>  = Vec::new();
        let mut second_idx : Vec <usize> = Vec::new();
        let mut mags : Vec <f64> = Vec::new();
        let mut dot_products : Vec <f64> = Vec::new();
        let mut mag_sum = 0_f64;

        let loop_start_time = now.elapsed().as_secs_f64();
        for (idx1, coordinate_1) in coordinates.data[0..coordinates.size-1].iter().enumerate() {
            for (idx2, coordinate_2) in coordinates.data[idx1+1..coordinates.size].iter().enumerate(){
                first_idx.push(idx1);
                second_idx.push(idx1+idx2+1);
                let difference = coordinate_1.sub(coordinate_2);
                mags.push(difference.mag);
                mag_sum += difference.mag;
                data.push(difference);
                dot_products.push(coordinate_1.dot(coordinate_2));
            }
        }
        unsafe{
            GLOB_NEW_COORDINATE_DIFFERENCES_LOOP_ONLY += now.elapsed().as_secs_f64() - loop_start_time;
        }

        let min_max_start_time = now.elapsed().as_secs_f64();
        let mean_magnitude = mag_sum  / (data.len() as f64);
        let delta_mags : Vec <f64> = mags.iter().map(|&x| x - mean_magnitude).collect(); //::<Vec<f64>>();
        let signs : Vec <f64> = mags.iter().map(|&x| (x - mean_magnitude)/(x - mean_magnitude).abs()).collect(); 
        let mut min_val : f64 = MAX_NUM;
        let mut max_val : f64 = MIN_NUM;

        for value in delta_mags.iter(){
            match *value < min_val { 
                true => {
                    min_val = *value;
                    // min_idx = idx;
                },
                false => ()
            }
            match *value > max_val { 
                true => {
                    max_val = *value;
                    // max_idx = idx;
                },
                false => ()
            }  
        }

        let mag_range = max_val - min_val;

        unsafe{
            GLOB_NEW_COORDINATE_DIFFERENCES_MIN_MAX_ONLY += now.elapsed().as_secs_f64() - min_max_start_time;
        }

        let result = CoordinateDifferences{

            size: data.len(),
            data,
            first_index : first_idx,
            second_index : second_idx,
            mean_magnitude,
            magnitude_range : mag_range,
            signs,
            dots : dot_products,
        };
        
        unsafe{
            GLOB_NEW_COORDINATE_DIFFERENCES += now.elapsed().as_secs_f64();
        }
        result
    }

    fn print(&self, precision: usize) {

        let field:usize = precision + 4; 
        const EPS: f64 = 1e-6;
        let mut angle_filter_count: usize = 0;

        colour::yellow_ln!("CoordinateDifferences::print()");
        colour::yellow_ln!("Length = {:3}",self.size);
        println!("E(|x|) = {:field$.precision$}",&self.mean_magnitude, precision=precision, field=field);
        for (idx1, idx2, coordinate, dot, sign) in izip!(&self.first_index, &self.second_index, &self.data, &self.dots, &self.signs) {
            colour::red!("({:3}, {:3}): ",idx1,idx2);
            coordinate.print(',',precision);
            colour::cyan!(" Sign = [{:+.0}]",sign);
            colour::cyan_ln!(" <a.b> = {:field$.precision$}",dot, field=field, precision=precision);
        }
        let edge_dots_and_unit_norms = CoordinateDifferences::get_edge_dots_and_unit_norms(&self.first_index,&self.second_index,&self.data);
        for edge_vec in edge_dots_and_unit_norms.iter() {
            colour::red!("<( {}, {}, {} ):",
                        edge_vec[0] as usize,
                        edge_vec[1] as usize,
                        edge_vec[2] as usize);
            colour::dark_green!(" unorm = ( {:field$.precision$}, {:field$.precision$}, {:field$.precision$})",
                        edge_vec[5],
                        edge_vec[6],
                        edge_vec[7],
                        field=field, precision=precision);
            colour::cyan_ln!(" unit dot = {:field$.precision$}, angle = {:field$.angle_precision$}",
                        edge_vec[3],
                        edge_vec[4], 
                        field=field, precision=precision,angle_precision = precision-2);
        
            // println!("<( {}, {}, {} ): unit dot = {:field$.precision$}, angle = {:field$.angle_precision$}, unorm = ( {:field$.precision$}, {:field$.precision$}, {:field$.precision$})",
            //             edge_vec[0],
            //             edge_vec[1],
            //             edge_vec[2],
            //             edge_vec[3],
            //             edge_vec[4], 
            //             edge_vec[5], 
            //             edge_vec[6], 
            //             edge_vec[7], 
            //             field=field, precision=precision,angle_precision = precision-2);
        
                    }

        println!("max(|x|) - min(|x|) = {:field$.precision$}",&self.magnitude_range, precision=precision, field=field);
    }

    fn edge_copy( edge: &[f64]) -> Vec<f64> {

        return edge.iter().copied().collect()
    }

    fn order_edges( first_edge: &[f64] ,  second_edge: &[f64]) -> EdgeSwapResult {
        if first_edge[0] < second_edge[0] { 
            EdgeSwapResult{edge1 : CoordinateDifferences::edge_copy(first_edge), edge2 : CoordinateDifferences::edge_copy(second_edge), swapped : false}
        } else if first_edge[0] > second_edge[0] {
            EdgeSwapResult{edge1 : CoordinateDifferences::edge_copy(second_edge), edge2 : CoordinateDifferences::edge_copy(first_edge), swapped : true}
        } else if first_edge[1] < second_edge[1] {
            EdgeSwapResult{edge1 : CoordinateDifferences::edge_copy(first_edge), edge2 : CoordinateDifferences::edge_copy(second_edge), swapped : false}
        } else if first_edge[1] > second_edge[1] {
            EdgeSwapResult{edge1 : CoordinateDifferences::edge_copy(second_edge), edge2 : CoordinateDifferences::edge_copy(first_edge), swapped : true}
        } else if first_edge[2] < second_edge[2] {
            EdgeSwapResult{edge1 : CoordinateDifferences::edge_copy(first_edge), edge2 : CoordinateDifferences::edge_copy(second_edge), swapped : false}
        } else {
            EdgeSwapResult{edge1 : CoordinateDifferences::edge_copy(second_edge), edge2 : CoordinateDifferences::edge_copy(first_edge), swapped : true}
        }
    }

    fn edge_vector_copy( edge_data: &[Vec<f64>]) -> Vec<Vec<f64>> {

        return edge_data.iter().map(|x| CoordinateDifferences::edge_copy(x)).collect()
    }

    
    fn sort_edge_dots(edge_data: &[Vec<f64>]) -> Vec<Vec<f64>> {

        let len: usize = edge_data.len();
        let mut sorted_edges = CoordinateDifferences::edge_vector_copy(edge_data);


        for _i in 0.. len - 1 { // bubble sort
            let mut swapped: bool = false;
            for j in 0.. len - 1 {
                let this_swap_result = CoordinateDifferences::order_edges(&sorted_edges[j], &sorted_edges[j+1]);
                sorted_edges[j] = this_swap_result.edge1;
                sorted_edges[j+1] = this_swap_result.edge2;
                swapped |= this_swap_result.swapped;
            }
            if ! swapped {
                break;
            }
        }
        sorted_edges
    }

    fn get_edge_dots_and_unit_norms(first_index:&[usize],second_index:&[usize],data:&[Coordinate]) -> Vec<Vec<f64>>{


        let mut result : Vec<Vec<f64>> = Vec::new();
        let len = first_index.len();
        for ((idx1, first_idx1), second_idx1) in izip!(first_index[..len-1].iter().enumerate(),second_index[..len-1].iter()){
            for ((idx2, first_idx2), second_idx2) in izip!(first_index.iter().enumerate(),second_index.iter()){
                if idx2 <= idx1 {
                    continue;
                }
                if *first_idx1 == *first_idx2 {
                    let mut this_vec : Vec<f64> = Vec::new();                 
                    this_vec.extend([*second_idx1 as f64, *first_idx1 as f64, *second_idx2 as f64, data[idx1].unit_dot(&data[idx2]), data[idx1].angle(&data[idx2]) ]); 
                    this_vec.extend(data[idx1].unit_cross(&data[idx2]).expand().iter());
                    result.push(this_vec);
                } else if second_idx1 == second_idx2 {
                        let mut this_vec : Vec<f64> = Vec::new();
                        this_vec.extend([*first_idx1 as f64, *second_idx1 as f64, *first_idx2 as f64, data[idx1].unit_dot(&data[idx2]), data[idx1].angle(&data[idx2])]); 
                        this_vec.extend(data[idx1].unit_cross(&data[idx2]).expand().iter());
                        result.push(this_vec);
                } else if first_idx1 == second_idx2 {
                        let mut this_vec : Vec<f64> = Vec::new();
                        this_vec.extend([*second_idx1 as f64, *first_idx1 as f64, *first_idx2 as f64, -1.0 * data[idx1].unit_dot(&data[idx2]), data[idx1].angle(&data[idx2].mult(-1.0))]); 
                        this_vec.extend(data[idx1].unit_cross(&data[idx2]).expand().iter());
                        result.push(this_vec);
                } else if second_idx1 == first_idx2 {
                    let mut this_vec : Vec<f64> = Vec::new();
                    this_vec.extend([*first_idx1 as f64, *second_idx1 as f64, *second_idx2 as f64, -1.0 * data[idx1].unit_dot(&data[idx2]), data[idx1].angle(&data[idx2].mult(-1.0))]); 
                    this_vec.extend(data[idx1].unit_cross(&data[idx2]).expand().iter());
                    result.push(this_vec);
                }
            }
        } 
        CoordinateDifferences::sort_edge_dots(&result)
    }

}

#[derive(Debug)]
struct UnitNormSingle {
    vertices : Vec<usize>,
    unit_norm : Coordinate,
    unit_dot : f64,
    angle: f64,
    face_index: usize,
    // unit_norms_index: usize,
}

impl UnitNormSingle {

    fn new( vertices : &[usize], unit_norm: & Coordinate, unit_dot: f64, angle: f64, face_index: usize) //, unit_norms_index: usize) 
                ->UnitNormSingle {

        UnitNormSingle {
            vertices: vertices.iter().copied().collect(),
            unit_norm: unit_norm.copy(),
            unit_dot, 
            angle,
            face_index,
            // unit_norms_index,
        }
    }

    fn copy(self: & UnitNormSingle) -> UnitNormSingle {

        UnitNormSingle::new(&self.vertices, &self.unit_norm, self.unit_dot, self.angle, self.face_index) //, self.unit_norms_index)

    }
}


#[derive(Debug)]
struct UnitNormSwapResult {

    unit_norms_pair : (UnitNormSingle, UnitNormSingle),
    indices: (usize, usize),
    swapped: bool,
}

impl UnitNormSwapResult {

    fn new(un: &UnitNorms, index: usize) -> UnitNormSwapResult {

        if index < un.size - 1 {
            let unit_norms_pair = (un.get_indexed_unit_norm(index), un.get_indexed_unit_norm(index + 1));
            let swapped = false;
            UnitNormSwapResult {

                unit_norms_pair,
                indices: (index, index+1),
                swapped,
            }

        } else {
            panic!("UnitSwapResult::new() called with an index of {}, the size of unit_norms is only {}.\n\
            A valid index required that 0 < {} < {}",index,un.size,index,un.size-1)
        }
        

    }

    fn copy(&self) ->  UnitNormSwapResult {
        UnitNormSwapResult {

            unit_norms_pair: (self.unit_norms_pair.0.copy(), self.unit_norms_pair.1.copy()),
            indices: (self.indices.0, self.indices.1),
            swapped: self.swapped,
        }
    }

    fn new_from_unit_norm_singles(uns0: &UnitNormSingle, uns1: &UnitNormSingle, indices: (usize, usize), swapped: bool) -> UnitNormSwapResult {

        UnitNormSwapResult {

            unit_norms_pair: (uns0.copy(), uns1.copy()),
            indices,
            swapped,
        }
    }

    fn comp(self : &UnitNormSwapResult) -> bool { // calculate whether the first face index is less than the second face index
        
        self.unit_norms_pair.0.face_index <= self.unit_norms_pair.1.face_index
    }

    fn swap( self: &UnitNormSwapResult) -> UnitNormSwapResult { 

        let index1 = self.indices.0;
        let index2 = self.indices.1;

        let swapped = true;
        UnitNormSwapResult::new_from_unit_norm_singles(&self.unit_norms_pair.1,
                                                        &self.unit_norms_pair.0, 
                                                        (index1, index2), 
                                                        swapped)
    }

    fn ordered_unit_norm(self : & UnitNormSwapResult) -> UnitNormSwapResult {

        if self.comp() {
            self.copy()
        } else {
            self.swap()
        }
    }

    fn print(self: & UnitNormSwapResult){

        let precision: usize = 5;
        let field: usize = 10;

        colour::yellow_ln!("UnitNormSwapResult::print()");
        for (idx,&unp) in [&self.unit_norms_pair.0,&self.unit_norms_pair.1].iter().enumerate() {
            println!("{} : Face# {} : vertices ∠( {}, {}, {} ): unit dot = {:field$.precision$}, angle = {:field$.angle_precision$}, unorm = ( {:field$.precision$}, {:field$.precision$}, {:field$.precision$})",
            // indices are stored in the swap result to avoid confusion
            match idx {
                0 => self.indices.0,
                1 => self.indices.1,
                _ => panic!("Too many indices for print in UnitNormSwapResult"), // this should not be needed, only here to keep compiler happy
            },
            unp.face_index,
            unp.vertices[0],
            unp.vertices[1],
            unp.vertices[2],
            unp.unit_dot,
            unp.angle, 
            unp.unit_norm.x, 
            unp.unit_norm.y,
            unp.unit_norm.z, 
            field=field, precision=precision,angle_precision = precision-2);
        }

    }

        
}

#[derive(Debug)]
struct Faces {
    vertices : Vec<Vec<usize>>,
    unit_norms : CoordinateVector,
    face_indices: Vec<usize>,
    unique_face_indices: Vec<usize>,
    centroids : CoordinateVector,
    coordinates : CoordinateVector,
    size: usize,
    num_camera_angles: usize,
}

impl Faces {

    fn new( un : & UnitNorms) -> Faces {

        let len= un.size;
        let mut sorted_unit_norms = un.copy();


        

        // let mut new_face_count = unique_face_indices[unique_face_indices.len() - 1];
        // let mut new_face_vertices: Vec<Vec<usize>> = Vec::new();

        // sorted_unit_norms.print(6);
        colour::red_ln!("{}",un.size);
        for _i in 0.. len - 1 { // bubble sort, note that using `len` will panic, must be len - 1
            let mut swapped: bool = false;
            for j in 0.. len - 1 {
                let this_swap_result = UnitNormSwapResult::new(&sorted_unit_norms,j);
                let ordered_swap_result = this_swap_result.ordered_unit_norm();
                swapped |= ordered_swap_result.swapped;
                sorted_unit_norms.in_place_store(&ordered_swap_result);
            }
            if ! swapped {
                break;
            }
            
        }



        let mut face_vertices= vec_2d_copy_usize(& sorted_unit_norms.vertices);
        let mut face_indices_copy= vec_copy_usize(& sorted_unit_norms.face_indices);
        let mut unit_norms_copy = sorted_unit_norms.unit_norms.copy();
        

        // println!("###################################################################################################
        // ###################################################################################################
        // ###################################################################################################
        // ###################################################################################################
        // ###################################################################################################
        // ###################################################################################################
        // ###################################################################################################");

        let mut insert = true;
        while insert  {
            insert = false;
            let mut removed_indices: Vec<usize> = Vec::new();
            let mut temp_vertices = vec_2d_copy_usize( &face_vertices);
            for i in 0..face_vertices.len()  - 1 {
                let face_index = face_indices_copy[i];
                if !removed_indices.is_empty() && removed_indices.contains(&i) {
                    continue;
                }
                for j in i+1..face_vertices.len() {
                    let this_face_index = face_indices_copy[j];
                    if face_index != this_face_index {
                        continue
                    }
                    if removed_indices.is_empty() | ! (removed_indices.contains(&j)) && Faces::in_plane(&face_vertices[i],&face_vertices[j]) {
                        Faces::insert_vertices(& mut temp_vertices, i, j);
                        removed_indices.push(j);
                        insert = true;
                    }
                }
            } 
            if insert {
                removed_indices.sort_unstable();
                removed_indices.reverse();
                let (fv, fic, unc) = 
                        Faces::reduce_planar_vertices(&temp_vertices, &face_indices_copy, &unit_norms_copy, &removed_indices);
                face_vertices = fv;
                face_indices_copy = fic;
                unit_norms_copy = unc;
            }
        }

        // println!("{:?}",face_vertices);
        // println!("###################################################################################################
        // ###################################################################################################
        // ###################################################################################################
        // ###################################################################################################
        // ###################################################################################################
        // ###################################################################################################
        // ###################################################################################################");

        let unique_face_indices:Vec<usize> = uniquely_sorted(&face_indices_copy);

        let size = face_indices_copy.len();

        face_indices_copy = (0..face_indices_copy.len()).collect();

        let centroids: CoordinateVector = Faces::calculate_centroids(& face_vertices, & un.coordinates, size);
    
        
        Faces {
            vertices : face_vertices,
            unit_norms : unit_norms_copy,
            face_indices: face_indices_copy,
            unique_face_indices,
            centroids,
            coordinates: un.coordinates.copy(),
            size,
            num_camera_angles: un.num_camera_angles,
        }
    }

    fn calculate_centroids(vertices: &[Vec<usize>], coordinates: &CoordinateVector, size: usize) -> CoordinateVector {

        let mut centroids = CoordinateVector::new_from_empty();
        for these_vertices in vertices.iter().take(size) {
            let mut centroid = Coordinate::zero();
            let vertex_count = these_vertices.len();
            for vertex in vec_copy_usize(these_vertices) {
                centroid = centroid.add(&coordinates.indexed_coordinate(vertex));
            }
            let centroid = centroid.mult( 1.0 / (vertex_count as f64));
            centroids.push(&centroid.copy());
        }
        centroids
    }

    

    fn insert_vertices(all_vertices: & mut Vec<Vec<usize>>, i:usize, j:usize)  {


        let mut face_vertices = vec_copy_usize(&all_vertices[i]);
        let mut these_vertices = vec_copy_usize(&all_vertices[j]);
        face_vertices.append(& mut these_vertices);
        face_vertices = uniquely_sorted(&face_vertices);
        all_vertices[i] = face_vertices;
    }
    
    fn in_plane(planar_vertices: &[usize], these_vertices: &[usize]) -> bool {

        if planar_vertices.is_empty() {
            return true;
        }
        for vertex in these_vertices {
            if planar_vertices.contains(vertex) {
                return true;
            }
        }
        false
    }
    fn reduce_planar_vertices(  planar_vertices: &[Vec<usize>], 
                                face_indices: &[usize], 
                                unit_normals: & CoordinateVector,
                                removed_indices: &[usize]) -> (Vec<Vec<usize>>, Vec<usize>, CoordinateVector) {

        let mut reduced_vertices : Vec<Vec<usize>> = vec_2d_copy_usize(planar_vertices);
        let mut reduced_face_indices : Vec<usize> = vec_copy_usize(face_indices);
        let mut reduced_norms: CoordinateVector = unit_normals.copy();
        // println!("removed_indices {:?}",removed_indices);
        for &i in removed_indices{
            reduced_vertices.remove(i);
            reduced_face_indices.remove(i);
            reduced_norms.remove(i);
        }
        // println!("reduced_vertices{:?}",reduced_vertices);
        (reduced_vertices, reduced_face_indices, reduced_norms)
    }

    fn get_number_of_faces_of_size( & self) -> (Vec<usize> , Vec<usize>) {
        
        let mut sizes:Vec<usize> = Vec::new();
        let mut number_of_faces:Vec<usize> = Vec::new();

        for v in &self.vertices {
            let size = v.len(); // number of vertices in this face
            let opt_pos = sizes.iter().position(|&x| x == size);
            match opt_pos {
                Some(idx) => // this size is already in the sizes vector
                    number_of_faces[idx] += 1, // increment the face count
                None =>  {// this size is not in the sizes vector 
                    sizes.push(size); // add to sizes vector
                    number_of_faces.push(1); // initialise the face count to 1
                }
            }
        }
            
        (sizes, number_of_faces)
    }

    fn print(&self, precision: usize) {

        let field:usize = precision + 4;


        colour::yellow_ln!("Faces::print()");

        for idx in 0..self.size{
            colour::dark_yellow!("{} : ", idx);
            colour::red!("Face# {}: vertices=", &self.face_indices[idx]);
            vec_usize_print(&self.vertices[idx]);
            colour::dark_green!(" unorm = ( {:field$.precision$}, {:field$.precision$}, {:field$.precision$})",
                self.unit_norms.data[idx].x, self.unit_norms.data[idx].y, self.unit_norms.data[idx].z,field=field, precision=precision);
            colour::red!(" centroids = ( {:field$.precision$}, {:field$.precision$}, {:field$.precision$})",
                self.centroids.data[idx].x, self.centroids.data[idx].y, self.centroids.data[idx].z,field=field, precision=precision);
            let u_centroid = self.centroids.data[idx].make_unit_vector();
            let n_dot_c = self.unit_norms.data[idx].dot(& u_centroid);
            colour::dark_cyan_ln!(" <unorm,ucentroid> = {:field$.precision$}",
            n_dot_c,field=field, precision=precision);
        }

        let (sizes, number_of_faces) = self.get_number_of_faces_of_size();
        println!("");
        for idx in 0..sizes.len() {
            colour::blue_ln!("Vertices in face = {}, number of faces = {}", sizes[idx],number_of_faces[idx]);
        }
    }

    fn check_outer_face(& self,  face_index_camera: usize) {

        let camera_direction = self.centroids.indexed_coordinate(face_index_camera);

        let mut max_d = 0.0;
        let mut max_index:usize = 0;
        
        for idx in 0..self.size {
            let unorm_plane = self.unit_norms.indexed_coordinate(idx);
            let plane_centroid = self.centroids.indexed_coordinate(idx);
            let ci_dot_n = plane_centroid.dot(&unorm_plane);
            let cj_dot_n = camera_direction.dot(&unorm_plane);
            let alpha = ci_dot_n /cj_dot_n ;
    
            let d = alpha * camera_direction.mag; 
            let cj_intersection = camera_direction.mult(alpha);
            // let mut polygon: Polygon<f64> = Polygon::new(LineString::from(vec![(0.,0.)]),
            //                             vec![],);
            let mut xy_vertices: CoordinateVector = CoordinateVector::new_from_empty();
            let mut vertices: CoordinateVector = CoordinateVector::new_from_empty();
            if cj_dot_n.abs() > 0.001{
                for &i in &self.vertices[idx] {
                    // println!("Projected coordinates, idx = {}",idx);
                    let xy_vertex = self.coordinates.indexed_coordinate(i).project_onto_xy();
                    let vertex = self.coordinates.indexed_coordinate(i);
                    xy_vertices.push(&xy_vertex);
                    vertices.push(&vertex);
                    vertex.print('\n',5);
                }
                // polygon = xy_vertices.project_to_xy_onto_geo_polygon(GEO_SCALE);
                let xy_intersection = cj_intersection.project_onto_xy();
                colour::yellow_ln!("Point under examination, xy projection:");
                xy_intersection.print('\n',5);
                // let point = xy_intersection.project_to_xy_convert_to_geo_coordinate(GEO_SCALE);
                let my_polygon_contains = xy_vertices.polygon_xy_contains(&xy_intersection);
                // if polygon.contains(&point) {
                if my_polygon_contains {
                    colour::green_ln!("{} included in polygon {}",idx, face_index_camera);
                    if d > max_d {
                        max_d = d;
                        max_index = idx;
                        // println!("MAX!")
                    }
                    colour::green_ln!("distance = {}",d);
                } else {
                    colour::red_ln!("{} excluded from polygon {}", idx, face_index_camera);
                }
            }
        }
        colour::cyan_ln!("Outer most face for {} is {}", face_index_camera,max_index);

        
        
        
        
        
        



        // colour::magenta!("ni = unorm[{}]: ",face_index_plane);
        // unorm_plane.print('\n',5);
        // colour::magenta!("ci = centroid[{}]: ",face_index_plane);
        // plane_centroid.print('\n',5);
        // colour::magenta!("cj = camera (centroid[{}]) dir: ",face_index_camera);
        // camera_direction.print('\n',5);
        // colour::blue_ln!("<ci,n> = {:0.5}, <cj,n> = {:0.5}",ci_dot_n,cj_dot_n);
        // colour::blue_ln!("alpha = {:0.5}, d = {:0.5}",alpha,d);


        // let cj_intersection = camera_direction.mult(alpha);
        // colour::magenta!("cj_intersection: ");
        // cj_intersection.print('\n',5);  
        
        // let acj_min_ci = camera_direction.mult(alpha).sub(&plane_centroid);
        // let acj_min_ci_dot_ni = acj_min_ci.dot(&unorm_plane);


        // colour::yellow!("acj_min_ci : ");
        // acj_min_ci.print('\n',5);

        // colour::yellow_ln!("acj_min_ci_dot_ni = {:0.10}",acj_min_ci_dot_ni);

        // let p_min_ci = cj_intersection.sub(&plane_centroid);
        // let p_min_ci_dot_ni = p_min_ci.dot(&unorm_plane);

        // colour::cyan!("p_min_ci_dot_ni = {:0.12}",p_min_ci_dot_ni);

        // let mut xy_vertices: CoordinateVector = CoordinateVector::new_from_empty();
        
        // println!("---------------------------------------------------------");
        // colour::red_ln!("Polygon surface, xy projection:");
        // let mut polygon: Polygon<f64> = Polygon::new(LineString::from(vec![(0.,0.)]),
        // vec![],);
        // if cj_dot_n.abs() > 0.1 {
        //     for &i in &self.vertices[face_index_plane] {
        //         let vertex = self.coordinates.indexed_coordinate(i).project_onto_xy();
        //         xy_vertices.push(&vertex);
        //         vertex.print('\n',5);
        //     }
        //     polygon = xy_vertices.project_to_xy_onto_geo_polygon();

        // } 
        // let xy_intersection = cj_intersection.project_onto_xy();
        // colour::red_ln!("Point under examination, xy projection:");
        // xy_intersection.print('\n',5);
        // let point = xy_intersection.project_to_xy_convert_to_geo_coordinate();
        // if polygon.contains(&point) {
        //     colour::green_ln!("Included in polygon");
        // } else {
        //     colour::red_ln!("Excluded from polygon");
        // }

    }
    // struct Faces {
    //     vertices : Vec<Vec<usize>>,
    //     unit_norms : CoordinateVector,
    //     face_indices: Vec<usize>,
    //     unique_face_indices: Vec<usize>,
    //     centroids : CoordinateVector,
    //     coordinates : CoordinateVector,
    //     size: usize,
    //     num_camera_angles: usize,
    // }

    fn copy (& self) -> Faces {

        Faces {
            vertices : vec_2d_copy_usize(&self.vertices),
            unit_norms : self.unit_norms.copy(),
            face_indices : vec_copy_usize(&self.face_indices),
            unique_face_indices : vec_copy_usize(&self.unique_face_indices),
            centroids : self.centroids.copy(),
            coordinates : self.coordinates.copy(),
            size : self.size,
            num_camera_angles : self.num_camera_angles,
        }

    }

    fn full_remove( & self, faces_to_remove: &[&usize]) -> Faces {

        // faces_to_remove must already be sorted in reverse order
        // otherwise non-sensical results will occur

        let mut result = self.copy();
        for &i in faces_to_remove {
            result.vertices.remove(*i);
            result.unit_norms.remove(*i);
            result.face_indices.remove(*i);
            result.centroids.remove(*i);
            result.size -= 1;
        }
        // Reindex the face_indices
        // result.face_indices = (0..result.size).collect();
        // result.unique_face_indices = (0..result.size).collect();
        result
    }

    fn get_outer_faces( & self) -> Faces{

        // let mut outer_faces: HashSet <usize>  = (0..self.size).collect();
        let mut outer_faces: HashSet <usize>  = HashSet::new();

        colour::magenta_ln!("Number of faces before get_outer_faces = {}",self.size);
        for sa in GoldenSpiral::new(self.num_camera_angles) {
            let camera_direction = Coordinate::new_from_spherical_coordinates(1.0,sa.theta,sa.phi);
            if self.coordinates.aligned_with_coordinate(&camera_direction) {
                continue;
            }
            let mut max_d = 0.0;
            let mut max_index: usize = 0;
            for idx2 in 0..self.size {

                let unorm_plane = self.unit_norms.indexed_coordinate(idx2);
                let plane_centroid = self.centroids.indexed_coordinate(idx2);
                let ci_dot_n = plane_centroid.dot(&unorm_plane);
                let cj_dot_n = camera_direction.dot(&unorm_plane);
                let alpha = ci_dot_n /cj_dot_n ;
        
                let d = alpha * camera_direction.mag; 
                let cj_intersection = camera_direction.mult(alpha);
                // let mut polygon: Polygon<f64> = Polygon::new(LineString::from(vec![(0.,0.)]),
                                            // vec![],);
                let mut xy_vertices: CoordinateVector = CoordinateVector::new_from_empty();
                if cj_dot_n.abs() > 0.001{
                    for &i in &self.vertices[idx2] {
                        // println!("Projected coordinates, idx = {}",idx);
                        let vertex = self.coordinates.indexed_coordinate(i).project_onto_xy();
                        xy_vertices.push(&vertex);
                        // vertex.print('\n',5);
                    }
                    // polygon = xy_vertices.project_to_xy_onto_geo_polygon(GEO_SCALE);
                    
                    let xy_intersection = cj_intersection.project_onto_xy();
                    // colour::yellow_ln!("Point under examination, xy projection:");
                    // xy_intersection.print('\n',5);
                    // let point = xy_intersection.project_to_xy_convert_to_geo_coordinate(GEO_SCALE);
                    let my_polygon_contains = xy_vertices.polygon_xy_contains(&xy_intersection);

                    if my_polygon_contains {

                    // if polygon.contains(&point) {
                        // colour::green_ln!("{} included in polygon {}",idx, face_index_camera);
                        if d > max_d {
                            max_d = d;
                            max_index = idx2;
                            // println!("MAX!")
                        }
                        // colour::green_ln!("distance = {}",d);
                    } 
                }
            }
            if max_d > 0.1 && outer_faces.insert(max_index) {
                colour::green_ln!("Adding outer face with index {}",max_index);
            }
            
        }
        let mut vec_outer_faces: Vec<usize> = outer_faces.iter().copied().collect();
        vec_outer_faces.sort_unstable();
        colour::blue_ln!("{:?}", vec_outer_faces);

        let all_faces:HashSet<usize> = (0..self.size).collect();

        let removed_faces = all_faces.difference(&outer_faces);
        let mut vec_faces_to_remove: Vec<&usize> = removed_faces.collect();
        vec_faces_to_remove.sort();
        vec_faces_to_remove.reverse();

        let faces = self.full_remove(& vec_faces_to_remove);
        colour::magenta_ln!("Number of faces after get_outer_faces = {}",faces.size);
        faces
    }

        // struct Faces {
    //     vertices : Vec<Vec<usize>>,
    //     unit_norms : CoordinateVector,
    //     face_indices: Vec<usize>,
    //     unique_face_indices: Vec<usize>,
    //     centroids : CoordinateVector,
    //     coordinates : CoordinateVector,
    //     size: usize,
    // }

}



#[derive(Debug)]
struct UnitNorms {
    vertices : Vec<Vec<usize>>,
    unit_norms : CoordinateVector,
    unit_dots : Vec<f64>,
    angles: Vec<f64>,
    face_indices: Vec<usize>,
    coordinates : CoordinateVector,
    size: usize,
    num_camera_angles: usize,
    // unique_face_indices: Vec<usize>,
}

impl UnitNorms {


    fn new(edge_dots_and_unit_norms: &[Vec<f64>], coordinates: &CoordinateVector, num_camera_angles: usize) -> UnitNorms{

        let mut vertices: Vec<Vec<usize>> = Vec::new();
        let mut unit_norms: CoordinateVector = CoordinateVector::new_from_empty();
        let mut unit_dots: Vec<f64> = Vec::new();
        let mut angles: Vec<f64> = Vec::new();
        let mut size: usize = 0;


        for dot_and_norm in edge_dots_and_unit_norms.iter(){

            let mut these_vertices : Vec<usize> = Vec::new();
            for &dni in dot_and_norm.iter().take(3) {
                these_vertices.push(dni as usize);
            }
            vertices.push(these_vertices);
            unit_dots.push(dot_and_norm[3]);
            angles.push(dot_and_norm[4]);

            unit_norms.push( & Coordinate::new_from_vector(dot_and_norm[5..8].to_vec()));
            size += 1;
        }

        let face_indices = UnitNorms::calc_face_indices(&unit_norms);

        let result = UnitNorms {
            vertices,
            unit_norms,
            unit_dots,
            angles,
            face_indices,
            coordinates: coordinates.copy(),
            size,
            num_camera_angles,
            // unique_face_indices,
        };
        result.remove_parallel_vertices(coordinates)

    }

    fn copy( &self) -> UnitNorms {


        UnitNorms {
            vertices: vec_2d_copy_usize(&self.vertices),
            unit_norms: self.unit_norms.copy(),
            unit_dots: vec_copy_f64(&self.unit_dots),
            angles: vec_copy_f64(&self.angles),
            face_indices: vec_copy_usize(&self.face_indices),
            coordinates: self.coordinates.copy(),
            size: self.size,
            num_camera_angles: self.num_camera_angles,
            // unique_face_indices:vec_copy_usize(&self.unique_face_indices),
        }

    }

    fn in_place_store(&mut self,  unsr: & UnitNormSwapResult) { 

        let first_index = unsr.indices.0;
        let second_index = unsr.indices.1;

        self.vertices[first_index] = vec_copy_usize(&unsr.unit_norms_pair.0.vertices);
        self.unit_norms.data[first_index] = unsr.unit_norms_pair.0.unit_norm.copy();
        self.unit_dots[first_index] = unsr.unit_norms_pair.0.unit_dot;
        self.angles[first_index] = unsr.unit_norms_pair.0.angle;
        self.face_indices[first_index] = unsr.unit_norms_pair.0.face_index;

        self.vertices[second_index] = vec_copy_usize(&unsr.unit_norms_pair.1.vertices);
        self.unit_norms.data[second_index] = unsr.unit_norms_pair.1.unit_norm.copy();
        self.unit_dots[second_index] = unsr.unit_norms_pair.1.unit_dot;
        self.angles[second_index] = unsr.unit_norms_pair.1.angle;
        self.face_indices[second_index] = unsr.unit_norms_pair.1.face_index;
    }

    fn remove_parallel_vertices( & self, coordinates: &CoordinateVector ) -> UnitNorms {

        let mut result = self.copy();
        let mut indices_to_remove: Vec<usize> = vec![];
        let mut ready_to_break = false;

        for idx in 0..result.size {
            // vec_usize_print(&result.vertices[idx]);
            // println!("");
            for idx1 in 0..2_usize {
                let vertex1 = result.vertices[idx][idx1];
                for idx2 in idx1 + 1..3_usize {
                    let vertex2 = result.vertices[idx][idx2];
                    let dotval = coordinates.data[vertex1].unit_dot(&coordinates.data[vertex2]).abs();
                    // println!("|{}.{}|={:0.6}",vertex1,vertex2,dotval);
                    if float_equals(dotval,1.0,1.0e-6) {
                        indices_to_remove.push(idx);
                        // println!("******* removing index {}",idx);
                        ready_to_break = true;
                        break;
                    }
                }
                if ready_to_break {
                    ready_to_break = false;
                    break;
                }
            }
        }
        indices_to_remove.sort_unstable();
        indices_to_remove.reverse();

        for idx in indices_to_remove {
            result.remove(idx);
        }

        result
    }



    fn calc_face_indices(un: & CoordinateVector) -> Vec<usize> { //}, Vec<usize>){

        let mut face_indices: Vec<i32> = vec![-1;un.size];

        let mut face_index: i32 = 0;
        // colour::magenta_ln!("un before calculating faces");
        // un.print(4);
        for (idx1, norm1) in un.data.iter().enumerate() {
            // colour::dark_blue_ln!("idx1 = {}, face_indices[idx1]= {}", idx1, face_indices[idx1]);
            if face_indices[idx1] < 0 {
                face_index += 1;
                face_indices[idx1] = face_index as i32;
                // colour::magenta!("Face index = {}, idx1 = {}, norm1 = ",face_index,idx1);
                // norm1.print('\n', 6);


                if idx1 < un.size - 1 {
                    for (idx2, norm2)  in un.data[idx1 + 1..un.size].iter().enumerate(){
                        if face_indices[idx2 + idx1 + 1] < 0 && norm1.equal_or_inverted(norm2) {
                            // colour::magenta!("**Face index = {}, idx1 = {}, idx2 + idx1 + 1 = {}, norm1 = ", face_index, idx1, idx2 + idx1 + 1);
                            // norm1.print('\n', 6);
                            // colour::magenta!("**Matching to norm2 = ");
                            // norm2.print('\n',6);
                            face_indices[idx2 + idx1 + 1] = face_index as i32;
                        }
                    }
                }
            }
            
        }

        let face_indices:Vec<usize>=face_indices.iter().map(|&x| x as usize).collect();

        // let unique_face_indices:Vec<usize> = uniquely_sorted(&face_indices);


        // return face_indices.iter().map(|&x| x as usize).collect();
        face_indices //, unique_face_indices);


    }

    fn get_indexed_unit_norm(& self, index: usize) -> UnitNormSingle {


        UnitNormSingle::new(&self.vertices[index],
                            &self.unit_norms.indexed_coordinate(index), 
                            self.unit_dots[index], 
                            self.angles[index], 
                            self.face_indices[index])
    }

    fn remove( & mut self, index: usize) {
        if index < self.size {
            self.vertices.remove(index);
            self.unit_norms.remove(index);
            self.unit_dots.remove(index);
            self.angles.remove(index);
            self.face_indices.remove(index);
            self.size -= 1;
        } else {
            panic!("Index out of bounds, requesting index {} and maximum index is {}",index,self.size);
        }
    }

  

    fn print(&self, precision: usize) {

        let field:usize = precision + 4;

        colour::yellow_ln!("UnitNorms::print()");
        for idx in 0..self.size{
            colour::dark_yellow!("{} : ",idx);
            colour::dark_red!("Face# {} : <( {}, {}, {} ): ",
            self.face_indices[idx],
            self.vertices[idx][0],
            self.vertices[idx][1],
            self.vertices[idx][2]);
            colour::dark_blue!("unit dot = {:field$.precision$}, angle = {:field$.angle_precision$}: ",
            self.unit_dots[idx],
            self.angles[idx], 
            field=field, precision=precision,angle_precision = precision-2);
            colour::dark_green_ln!("unorm = ( {:field$.precision$}, {:field$.precision$}, {:field$.precision$})",
            self.unit_norms.data[idx].x, 
            self.unit_norms.data[idx].y, 
            self.unit_norms.data[idx].z, 
            field=field, precision=precision);
        }
    }

}
#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone)]
enum InitialiseKey {
    Random,
    Fixed,
    Symmetric,
    GoldenSpiral,
}


fn main() {
    
    // Triangle, d3: 3 vertices, 1 face
    // Tetrahedron, d4: 4 vertices, 4 faces
    // Triangular bipyramid, d6 (not a cube): 5 vertices, 6 faces (at least with these initial values. This is not the only stable state.)
    // Equilateral octohedron, Square bipyramid, d8: 6 vertices (at each of the three axes)
    // Pentagonal bipyramid, d10: 7 vertices, 10 faces
    // Cube, d6: 8 vertices, 6 faces


    // Dodecahedron: d12 20 vertices, 12 faces
    // Icosahedron: d20 12 vertices, 20 faces

    let mut initialise_map: HashMap<InitialiseKey, char> = HashMap:: new();
    initialise_map.insert(InitialiseKey::Random,'r');
    initialise_map.insert(InitialiseKey::Fixed,'f');
    initialise_map.insert(InitialiseKey::Symmetric,'s');
    initialise_map.insert(InitialiseKey::GoldenSpiral,'g');

    let args: Vec<String> = env::args().collect();

    if args.len() <= 1 {
        println!("One argument is needed to specify the number of vertices which must be larger or equal to 3.");
        exit(1);
    }
    let mut opt_number_of_vertices:Option<usize> = None;
    let mut vertex_count_found = false;
    let mut initialisation_method: InitialiseKey = InitialiseKey::Symmetric;


    for arg in args.iter().take(args.len().min(3)).skip(1) {
        colour::red_ln!("arg = {}",arg);
        let mut this_arg_number =false;
        match arg.parse::<usize>() {
            Ok(value) => {
                opt_number_of_vertices = Some(value);
                vertex_count_found = true;
                this_arg_number = true;
            },
            Err(_e) => (),
        }
        if !this_arg_number {
            match arg.parse::<char>() {
                Ok(value) => {
                    for (&key, &v) in &initialise_map {
                        if v==value{
                            initialisation_method = key;
                            colour::magenta_ln!("Found key, character='{} and key = {:?}",v,key);
                        }
                    }
                },
                Err(_e) => (),
            }
        }
    }
    let number_of_vertices = opt_number_of_vertices.unwrap();
    if !vertex_count_found {
        panic!("Please enter the number of vertices in one of the first two command line arguments.")
    }
    if vertex_count_found && number_of_vertices < 3 {
        panic!("The number of vertices must be larger than or equal to 3, number_of_vertices={}.",number_of_vertices);
    }
    if number_of_vertices > 30 {
        panic!("You don't really want to run this with so many vertices, best to limit to a max of 30, number_of_vertices={}.",number_of_vertices);
    }
    colour::green_ln!("The number of vertices is {}, the initialisation method is {:?}", number_of_vertices, initialisation_method);

    let now = Instant::now();
    const SCALE : f64 =  0.1;
    const STOP_POWER : i32 = 10;
    let stop = 15_f64.powi(-STOP_POWER);
    const PRECISION: usize = STOP_POWER as usize + 1; 

    const CAMERA_ANGLES: usize = 1000;
    
    


    // let data = vec![x1,x2,x3,x4,x5,x6];
    let mut coordinates = match initialisation_method {
        InitialiseKey::Random => CoordinateVector::new_from_random_vertices(number_of_vertices),
        InitialiseKey::Fixed => CoordinateVector::new_from_fixed_sequence(number_of_vertices),
        InitialiseKey::Symmetric => CoordinateVector::new_from_symmetric_fixed_sequence(number_of_vertices),
        InitialiseKey::GoldenSpiral => CoordinateVector::new_from_golden_spiral(number_of_vertices),
    };
    
    
    
    // let mut new_coordinates = CoordinateVector::new(data);

    const NUMBER_OF_CYCLES_BETWEEN_PRINT: usize = 10000;
    

    let  scale = SCALE;
    let mut max_dx:f64 = 1e6;
    // let new_max_dx:f64 = 1e6;
    let mut counter: usize = 0;
    let mut count_a: f64 = 0.0;
    let mut count_b: f64 = 0.0;
    let mut count_c: f64 = 0.0;
    // let mut count_d: f64 = 0.0;
    let mut timing_inluding_prints: f64 = 0.0;
    let mut timing_exluding_prints: f64 = 0.0;
    let mut print_timer: f64 = 0.0;
    // let mut prev_max_distance: f64 = 1000.0;
    let print_sub_timer = now.elapsed().as_secs_f64();
    println!("Initial values **************************");
    coordinates.print(PRECISION);
    print_timer += now.elapsed().as_secs_f64() - print_sub_timer;

    let mut coordinate_differences = CoordinateDifferences::new(&coordinates);

    let print_sub_timer = now.elapsed().as_secs_f64();
    coordinate_differences.print(PRECISION);
    print_timer += now.elapsed().as_secs_f64() - print_sub_timer;
    loop {
        let cum_count = now.elapsed().as_secs_f64();
        // *************** Start of count_a section ****************
        let (new_coordinates, new_max_dx) = coordinates.reposition(&coordinate_differences, scale);
        if counter + 1 % NUMBER_OF_CYCLES_BETWEEN_PRINT == 0 {
            let print_sub_timer = now.elapsed().as_secs_f64();
            println!("loop: counter = {} **************************",counter);
            new_coordinates.print(PRECISION);
            print_timer += now.elapsed().as_secs_f64() - print_sub_timer;
        }

        let cum_count2 = now.elapsed().as_secs_f64();
        count_a += cum_count2 - cum_count;
        // *************** End of count_a section ****************

        // *************** Start of count_b section ****************

        let new_coordinate_differences = CoordinateDifferences::new(&new_coordinates);
        
        let cum_count = now.elapsed().as_secs_f64();
        count_b += cum_count - cum_count2;

        // *************** End of count_b section ****************

        // *************** Start of count_c section ****************

        if counter + 1 % NUMBER_OF_CYCLES_BETWEEN_PRINT == 0 {
            let print_sub_timer = now.elapsed().as_secs_f64();
            new_coordinate_differences.print(PRECISION);
            print_timer += now.elapsed().as_secs_f64() - print_sub_timer;
        }

        if counter + 1 % NUMBER_OF_CYCLES_BETWEEN_PRINT == 0 {
            let print_sub_timer = now.elapsed().as_secs_f64();
            println!("max_dx = {:.precision$}, new_max_dx = {:.precision$}, %diff = {:.precision$}",
                    max_dx, new_max_dx, 1. - new_max_dx/max_dx, precision = PRECISION+2);
            print_timer += now.elapsed().as_secs_f64() - print_sub_timer;
        }
        if new_max_dx < stop {
            let print_sub_timer = now.elapsed().as_secs_f64();
            let elapsed_time_excluding_output = now.elapsed().as_secs_f64();
            colour::yellow_ln!("Final values dx stop **************************");
            println!("loop: counter = {} **************************",counter);
            coordinates.print(PRECISION);
            coordinate_differences.print(PRECISION);
            println!("max_dx = {:.precision$}, new_max_dx = {:.precision$}, %diff = {:.precision$}",
                            max_dx, new_max_dx, 1. - new_max_dx/max_dx, precision = PRECISION+2);
            println!("Elpased time including final output ={:0.3} ms", 1000.0 * now.elapsed().as_secs_f64());
            println!("Elpased time excluding final output ={:0.3} ms", 1000.0 * elapsed_time_excluding_output);
            print_timer += now.elapsed().as_secs_f64() - print_sub_timer;
            break;
        }
 
        
        coordinates = new_coordinates;
        coordinate_differences = new_coordinate_differences;
        max_dx = new_max_dx;
        counter += 1;


        let cum_count2 = now.elapsed().as_secs_f64();
        count_c += cum_count2 - cum_count;

        // *************** End of count_c section ****************

        if counter > 10_usize.pow(6) {
            let print_sub_timer = now.elapsed().as_secs_f64();
            println!("Final values large count stop **************************");
            println!("loop: counter = {} **************************",counter);
            coordinates.print(PRECISION);
            coordinate_differences.print(PRECISION);
            println!("max_dx = {:.precision$}, new_max_dx = {:.precision$}, %diff = {:.precision$}",
                            max_dx, new_max_dx, 1. - new_max_dx/max_dx, precision = PRECISION+2);
            println!("Elpased time ={:0.3} ms", 1000.0 * now.elapsed().as_secs_f64());
            print_timer += now.elapsed().as_secs_f64() - print_sub_timer;
            break;
        }
    }
    let edge_dots_and_unit_norms = CoordinateDifferences::get_edge_dots_and_unit_norms(&coordinate_differences.first_index,&coordinate_differences.second_index,&coordinate_differences.data);
    let unit_norms: UnitNorms = UnitNorms::new(&edge_dots_and_unit_norms,&coordinates, CAMERA_ANGLES);
    // unit_norms.print(PRECISION);
    println!("*********************************************************************************************************************************");
    // let unit_norms = unit_norms.sort_by_face();
    let faces:Faces = Faces::new(&unit_norms);
    faces.print(5);
    // faces.check_outer_face(19);
    let faces = faces.get_outer_faces();
    faces.print(5);

    
    // unit_norms.print(PRECISION);

    // let print_sub_timer = now.elapsed().as_secs_f64();
    // println!("Number of loops = {}", counter);

    // println!("Time in count_a = {:0.6} ms", 1000.0 * count_a);
    // println!("Time in count_b = {:0.6} ms", 1000.0 * count_b);
    // println!("Time in count_c = {:0.6} ms", 1000.0 * count_c);
    // unsafe{
    //     println!("Time in CoordinateDifferences.new() = {:0.6} ms", 1000.0 * GLOB_NEW_COORDINATE_DIFFERENCES);
    //     println!("Time in CoordinateDifferences.new(), loop only = {:0.6} ms", 1000.0 * GLOB_NEW_COORDINATE_DIFFERENCES_LOOP_ONLY);
    //     println!("Time in CoordinateDifferences.new(), min max only = {:0.6} ms", 1000.0 * GLOB_NEW_COORDINATE_DIFFERENCES_MIN_MAX_ONLY);
    // }
    // let final_time = now.elapsed().as_secs_f64();
    // print_timer += final_time - print_sub_timer;
    // println!("Time spent printing to screen = {:0.6} ms", print_timer * 1000.0);
    // println!("Time not printing to screen = {:0.6} ms", (final_time - print_timer) * 1000.0);

}
