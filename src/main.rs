use itertools::{Itertools, izip};
use std::time::Instant;
// #[macro_use] extern crate text_io; // used to pause 
// use rand::prelude::*;
use rand::distributions::{Distribution, Uniform};
// use rand::SeedableRng;
// use rand::rngs::StdRng;
use std::f64::consts::PI;
use std::env;
use std::process::exit as exit;

const MAX_NUM : f64 = 1e6;
const MIN_NUM : f64 = 0.0;

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

fn vec_copy_usize( data: &Vec<usize>) -> Vec<usize> {

    data.into_iter().map(|&x| x).collect()
}

fn vec_2d_copy_usize( data: &Vec<Vec<usize>>) -> Vec<Vec<usize>> {

    let mut result: Vec<Vec<usize>> = Vec::new();


    for uvec in data.iter().map(|x| vec_copy_usize(&x)){
        result.push(uvec);
        // result.push(uvec.iter().map(|&x| x).collect());
    }

    result

}

fn vec_copy_f64( data: &Vec<f64>) -> Vec<f64> {

    data.into_iter().map(|&x| x).collect()
}


// fn vec_copy<T>(data : &Vec<T>) -> Vec<T> {
    
//     data.into_iter().map(|x| x).collect()
// }

// fn vec_2d_copy<T>( data: &Vec<Vec<T>>) -> Vec<Vec<T>> {

//     let mut result: Vec<Vec<T>> = Vec::new();
//     let mut uvec : Vec<T> = Vec::new();

//     for uvec in data.iter().map(|&x| x){
//         result.push(vec_copy(&uvec));
//         // result.push(uvec.iter().map(|&x| x).collect());
//     }

//     result

// }

#[derive(Debug)]
struct Coordinate {

    x: f64,
    y: f64,
    z: f64,
    mag: f64,
}



impl Coordinate {


    fn new(x : f64, y: f64, z: f64 ) -> Coordinate{

        let result = Coordinate {
            x, 
            y,
            z,
            mag : Coordinate::calc_mag(x,y,z),
        };

        result
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

        for idx in 0..size{
            if idx == 0 {
                x = coordinate[idx];
            }
            if idx == 1 {
                y = coordinate[idx];
            }
            if idx == 2 {
                z = coordinate[idx];
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

        let x:f64 = &self.x + &pt.x;
        let y:f64 = &self.y + &pt.y;
        let z:f64 = &self.z + &pt.z;


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
        
        let result = (x.powi(2)  + y.powi(2) + z.powi(2)).sqrt();

        result
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

        return self.sub(&pt).mag < eps

    }

    fn equal_or_inverted( &self, pt: &Coordinate) -> bool {     
        self.equal(&pt) | self.equal(&pt.mult(-1.0))
    }

    fn print(&self, ch:char, precision: usize) {

        let field:usize = precision + 4; 
    
        print!("( {:field$.precision$} , {:field$.precision$} , {:field$.precision$} ), |x| = {:field$.precision$}{}",
                    self.x,self.y,self.z, self.mag,ch,precision=precision,field=field)
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

    fn calculate_2d_determinant(a: &Coordinate, b: &Coordinate) -> f64 {
        // This function calculates the 2x2 determinant of vector a and b by
        // ignoring the z components.
        // If the z components are non-zero then the output is meaningless.
        if ! float_equals(a.z, 0.0, 1e-6) | ! float_equals(b.z, 0.0, 1e-6) {
            panic!("2D determinant called with non zero z values, a.z = {:0.8}, b.z = {:0.8}", a.z,b.z);
        }
        else {
            return a.x * b.y - a.y * b.x;
        }
    }

    fn calculate_determinant(a:&Coordinate, b:&Coordinate, c:&Coordinate) -> f64 {
        
        let mut a2d:Coordinate = a.clone();
        let mut b2d:Coordinate = b.clone();
        let mut c2d:Coordinate = c.clone();

        a2d.z = 0.0;
        b2d.z = 0.0;
        c2d.z = 0.0;

        return a.z * Coordinate::calculate_2d_determinant(&b2d, &c2d) 
            - b.z * Coordinate::calculate_2d_determinant(&a2d, &c2d) 
            + c.z * Coordinate::calculate_2d_determinant(&a2d, &b2d);

    }

}


#[derive(Debug)]
struct CoordinateVector {

    size: usize,
    data:  Vec<Coordinate>,
}

impl CoordinateVector {
    
    
    fn new(data : Vec<Coordinate> ) -> CoordinateVector{

        let result = CoordinateVector{

            size: data.len(),
            data, // note this should move data
        };
        result
    }




    fn new_from_empty() -> CoordinateVector {
        
        let mut data: Vec<Coordinate> = Vec::new();
        let mut size:usize = 0;

        CoordinateVector {
            size,
            data,
        }
    }

    fn new_from_random_vertices(vertices: usize) -> CoordinateVector {

        if vertices < 2 {
            panic!("Must have more than one vertex")
        } else {

            let mut data : Vec<Coordinate> = Vec::new(); 
            for _ in 0..vertices{
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
                    println!(" length = {}",data.len());
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


    fn clone(&self) -> CoordinateVector {

        let mut data:  Vec<Coordinate> = Vec::new();

        for coordinate in &self.data {
            data.push(coordinate.clone())
        }
        return CoordinateVector::new(data);
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
        self.size = self.size + 1;

    }

    fn print(&self, precision: usize) {


        println!("Length = {:3}",self.size);
        for (idx, coordinate) in self.data.iter().enumerate() {
            print!("{}:",idx);
            coordinate.print('\n', precision);
        }
    }

    fn max_mag(&self) -> f64 {
        self.clone().data.iter().map(|x| x.mag).fold(-1e6,f64::max) // https://stackoverflow.com/a/66455028/1542485
        
    }

    fn reposition (&self, differences: &CoordinateDifferences, scale:f64, counter: usize, number_of_cycles_between_print: usize) -> (CoordinateVector, f64) {

        let mut result = self.clone();
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
        // if counter % number_of_cycles_between_print == 0 { 
        //     println!("***************** dx_parallel *********************");
        //     dx_parallel.print(STOP_POWER as usize + 2);
        //     println!("dx_parallel.max_mag()={}", dx_parallel.max_mag());
        //     println!("***************** dx *********************");
        // }
        return (new_result, true_dx.max_mag());
    }

    fn indexed_coordinate(&self, index: usize) -> Coordinate{

        if index < self.size {
            return self.data[index].copy();
        } else {
            panic!("Index out of bounds, requesting index {} and maximum index is {}",index,self.size);
        }
        
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


// This is using up all the time!!!!!!!!!
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
        const FLOAT_FILTER: bool = false;
        const ANGLE_FILTER: f64 = 60.0;
        const EPS: f64 = 1e-6;
        let mut angle_filter_count: usize = 0;

        
        
        println!("Length = {:3}",self.size);
        println!("E(|x|) = {:field$.precision$}",&self.mean_magnitude, precision=precision, field=field);
        for (idx1, idx2, coordinate, dot, sign) in izip!(&self.first_index, &self.second_index, &self.data, &self.dots, &self.signs) {
            print!("({:3}, {:3}): ",idx1,idx2);
            coordinate.print(',',precision);
            print!(" Sign = [{:+.0}]",sign);
            println!(" <a.b> = {:field$.precision$}",dot, field=field, precision=precision);
        }
        if FLOAT_FILTER {
            println!("Vertices filtered by angle to {}",ANGLE_FILTER);
        }
        let edge_dots_and_unit_norms = CoordinateDifferences::get_edge_dots_and_unit_norms(&self.first_index,&self.second_index,&self.data);
        for edge_vec in edge_dots_and_unit_norms.iter() {
            if ! FLOAT_FILTER{
                println!("<( {}, {}, {} ): unit dot = {:field$.precision$}, angle = {:field$.angle_precision$}, unorm = ( {:field$.precision$}, {:field$.precision$}, {:field$.precision$})",
                            edge_vec[0],
                            edge_vec[1],
                            edge_vec[2],
                            edge_vec[3],
                            edge_vec[4], 
                            edge_vec[5], 
                            edge_vec[6], 
                            edge_vec[7], 
                            field=field, precision=precision,angle_precision = precision-2);
            } else {
                if float_equals(ANGLE_FILTER, edge_vec[4], EPS) {
                    angle_filter_count += 1; 
                    println!("{}, <( {}, {}, {} ): unit dot = {:field$.precision$}, angle = {:field$.angle_precision$}",angle_filter_count, edge_vec[0],edge_vec[1],edge_vec[2],edge_vec[3],edge_vec[4], field=field, precision=precision,angle_precision = precision-2);
                }
            }

        }
        if FLOAT_FILTER {
            println!("Total filtered vertices =  {}",angle_filter_count); 
        }
        println!("max(|x|) - min(|x|) = {:field$.precision$}",&self.magnitude_range, precision=precision, field=field);
    }

    fn edge_copy( edge: & Vec <f64>) -> Vec<f64> {

        return edge.iter().map(|x| *x).collect()
    }

    fn order_edges( first_edge: & Vec<f64> ,  second_edge: & Vec<f64>) -> EdgeSwapResult {
        if first_edge[0] < second_edge[0] { 
            return EdgeSwapResult{edge1 : CoordinateDifferences::edge_copy(first_edge), edge2 : CoordinateDifferences::edge_copy(second_edge), swapped : false};
        } else if first_edge[0] > second_edge[0] {
            return EdgeSwapResult{edge1 : CoordinateDifferences::edge_copy(second_edge), edge2 : CoordinateDifferences::edge_copy(first_edge), swapped : true};
        } else if first_edge[1] < second_edge[1] {
            return EdgeSwapResult{edge1 : CoordinateDifferences::edge_copy(first_edge), edge2 : CoordinateDifferences::edge_copy(second_edge), swapped : false};
        } else if first_edge[1] > second_edge[1] {
            return EdgeSwapResult{edge1 : CoordinateDifferences::edge_copy(second_edge), edge2 : CoordinateDifferences::edge_copy(first_edge), swapped : true};
        } else if first_edge[2] < second_edge[2] {
            return EdgeSwapResult{edge1 : CoordinateDifferences::edge_copy(first_edge), edge2 : CoordinateDifferences::edge_copy(second_edge), swapped : false};
        } else {
            return EdgeSwapResult{edge1 : CoordinateDifferences::edge_copy(second_edge), edge2 : CoordinateDifferences::edge_copy(first_edge), swapped : true};
        }
    }

    fn edge_vector_copy( edge_data: & Vec<Vec<f64>>) -> Vec<Vec<f64>> {

        return edge_data.iter().map(|x| CoordinateDifferences::edge_copy(x)).collect()
    }

    
    fn sort_edge_dots(edge_data: & Vec<Vec<f64>>) -> Vec<Vec<f64>> {

        let len: usize = edge_data.len();
        let mut sorted_edges = CoordinateDifferences::edge_vector_copy(edge_data);


        for _i in 0.. len - 1 { // bubble sort
            let mut swapped: bool = false;
            for j in 0.. len - 1 {
                let this_swap_result = CoordinateDifferences::order_edges(&sorted_edges[j], &sorted_edges[j+1]);
                sorted_edges[j] = this_swap_result.edge1;
                sorted_edges[j+1] = this_swap_result.edge2;
                swapped = swapped | this_swap_result.swapped;
            }
            if ! swapped {
                break;
            }
        }
        sorted_edges


    }

    fn get_edge_dots_and_unit_norms(first_index:&Vec <usize>,second_index:&Vec <usize>,data:&Vec<Coordinate>) -> Vec<Vec<f64>>{


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
    indices : Vec<usize>,
    unit_norm : Coordinate,
    unit_dot : f64,
    angle: f64,
    face_index: usize,
    unit_norms_index: usize,
}

impl UnitNormSingle {

    fn new( indices : & Vec<usize>, unit_norm: & Coordinate, unit_dot: f64, angle: f64, face_index: usize, unit_norms_index: usize) 
    
                ->UnitNormSingle {

        UnitNormSingle {
            indices: indices.iter().map(|&x| x).collect(),
            unit_norm: unit_norm.copy(),
            unit_dot, 
            angle,
            face_index,
            unit_norms_index,
        }
    }

    fn copy(self: & UnitNormSingle) -> UnitNormSingle {

        UnitNormSingle::new(&self.indices, &self.unit_norm, self.unit_dot, self.angle, self.face_index, self.unit_norms_index)

    }
}


#[derive(Debug)]
struct UnitNormSwapResult {

    unit_norms_pair : (UnitNormSingle, UnitNormSingle),
    swapped: bool,
}

impl UnitNormSwapResult {

    fn new(un: &UnitNorms, index: usize) -> UnitNormSwapResult {

        if index < un.size - 1 {
            let unit_norms_pair = (un.get_indexed_unit_norm(index), un.get_indexed_unit_norm(index + 1));
            let swapped = false;
            UnitNormSwapResult {

                unit_norms_pair,
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
            swapped: self.swapped,
        }
    }

    fn new_from_unit_norm_singles(uns0: &UnitNormSingle, uns1: &UnitNormSingle, swapped: bool) -> UnitNormSwapResult {

        UnitNormSwapResult {

            unit_norms_pair: (uns0.copy(), uns1.copy()),
            swapped,
        }
    }

    fn comp(self : &UnitNormSwapResult) -> bool { // calculate whether the first face index is less than the second face index
        
        self.unit_norms_pair.0.face_index <= self.unit_norms_pair.1.face_index
    }

    fn swap( self: &UnitNormSwapResult) -> UnitNormSwapResult { // The unit norms need swapping but so does the unit_norms_index!!

        let mut uns0: UnitNormSingle = self.unit_norms_pair.0.copy();
        let mut uns1: UnitNormSingle = self.unit_norms_pair.1.copy();
        let swapped = true;
        let first_index = uns0.unit_norms_index;
        let second_index = uns1.unit_norms_index;
        uns0.unit_norms_index = second_index;
        uns1.unit_norms_index = first_index;
        println!("Swapping indices {} <-> {}", first_index,second_index);

        UnitNormSwapResult::new_from_unit_norm_singles(&self.unit_norms_pair.1, &self.unit_norms_pair.0, swapped)

    }

    fn ordered_unit_norm(self : & UnitNormSwapResult) -> UnitNormSwapResult {

        if self.comp() {
            self.copy()
        } else {
            self.swap()
        }
    }
}

#[derive(Debug)]
struct UnitNorms {
    indices : Vec<Vec<usize>>,
    unit_norms : CoordinateVector,
    unit_dots : Vec<f64>,
    angles: Vec<f64>,
    face_indices: Vec<usize>,
    size: usize,
}

impl UnitNorms{


    fn new(edge_dots_and_unit_norms: & Vec<Vec<f64>>) -> UnitNorms{

        let mut indices: Vec<Vec<usize>> = Vec::new();
        let mut unit_norms: CoordinateVector = CoordinateVector::new_from_empty();
        let mut unit_dots: Vec<f64> = Vec::new();
        let mut angles: Vec<f64> = Vec::new();
        let mut size: usize = 0;

        for dot_and_norm in edge_dots_and_unit_norms.iter(){

            let mut these_indices : Vec<usize> = Vec::new();
            for idx1 in 0..3 {
                these_indices.push(dot_and_norm[idx1] as usize);
            }
            indices.push(these_indices);
            unit_dots.push(dot_and_norm[3]);
            angles.push(dot_and_norm[4]);

            unit_norms.push( & Coordinate::new_from_vector(dot_and_norm[5..8].to_vec()));
            size += 1;
        }

        let face_indices = UnitNorms::calc_face_indices(&unit_norms);

        UnitNorms {
            indices,
            unit_norms,
            unit_dots,
            angles,
            face_indices,
            size,
        }
    }

    fn copy( &self) -> UnitNorms {


        UnitNorms {
            indices: vec_2d_copy_usize(&self.indices),
            unit_norms: self.unit_norms.copy(),
            unit_dots: vec_copy_f64(&self.unit_dots),
            angles: vec_copy_f64(&self.angles),
            face_indices: vec_copy_usize(&self.face_indices),
            size: self.size,
        }

    }

    fn in_place_store(&mut self,  unsr: & UnitNormSwapResult) { 

        let first_index = unsr.unit_norms_pair.0.unit_norms_index;
        let second_index = unsr.unit_norms_pair.1.unit_norms_index;

        self.indices[first_index] = vec_copy_usize(&unsr.unit_norms_pair.0.indices);
        self.unit_norms.data[first_index] = unsr.unit_norms_pair.0.unit_norm.copy();
        self.unit_dots[first_index] = unsr.unit_norms_pair.0.unit_dot;
        self.angles[first_index] = unsr.unit_norms_pair.0.angle;
        self.face_indices[first_index] = unsr.unit_norms_pair.0.face_index;

        self.indices[second_index] = vec_copy_usize(&unsr.unit_norms_pair.1.indices);
        self.unit_norms.data[second_index] = unsr.unit_norms_pair.1.unit_norm.copy();
        self.unit_dots[second_index] = unsr.unit_norms_pair.1.unit_dot;
        self.angles[second_index] = unsr.unit_norms_pair.1.angle;
        self.face_indices[second_index] = unsr.unit_norms_pair.1.face_index;

    }

    fn calc_face_indices(un: & CoordinateVector) -> Vec<usize>{

        let mut face_indices: Vec<i32> = vec![-1;un.size];

        let mut face_index: i32 = 0;
        for (idx1, norm1) in un.data.iter().enumerate() {
            if face_indices[idx1] < 0 {
                face_index += 1;
                face_indices[idx1] = face_index as i32;
            }
            if idx1 < un.size - 1 {
                for (idx2, norm2)  in un.data[idx1 + 1..un.size].iter().enumerate(){
                    if face_indices[idx2 + idx1 + 1] < 0 {
                        if norm1.equal_or_inverted(norm2) {
                            face_indices[idx2 + idx1 + 1] = face_index as i32;
                        }
                    }
                }
            }
        }

        return face_indices.iter().map(|&x| x as usize).collect();

    }

    fn get_indexed_unit_norm(& self, index: usize) -> UnitNormSingle {


        UnitNormSingle::new(&self.indices[index],
                            &self.unit_norms.indexed_coordinate(index), 
                            self.unit_dots[index], 
                            self.angles[index], 
                            self.face_indices[index],
                            index)
    }

    fn sort_by_face(& self) -> UnitNorms {


        let len= self.size;
        let mut sorted_unit_norms = UnitNorms::copy(&self);
    
    
        for _i in 0.. len - 1 { // bubble sort, note that using `len` will panic, must be len - 1
            let mut swapped: bool = false;
            for j in 0.. len - 1 {
                let this_swap_result = UnitNormSwapResult::new(&self,j).ordered_unit_norm();
                swapped = swapped | this_swap_result.swapped;
                sorted_unit_norms.in_place_store(&this_swap_result);

            }
            if ! swapped {
                break;
            }
        }
        sorted_unit_norms
    }

    fn print(&self, precision: usize) {

        let field:usize = precision + 4;

        println!("{}",self.unit_norms.data[3].z);
        println!();

        for idx in 0..self.size{
            println!("{} : Face# {} : <( {}, {}, {} ): unit dot = {:field$.precision$}, angle = {:field$.angle_precision$}, unorm = ( {:field$.precision$}, {:field$.precision$}, {:field$.precision$})",
            idx,
            self.face_indices[idx],
            self.indices[idx][0],
            self.indices[idx][1],
            self.indices[idx][2],
            self.unit_dots[idx],
            self.angles[idx], 
            self.unit_norms.data[idx].x, 
            self.unit_norms.data[idx].y, 
            self.unit_norms.data[idx].z, 
            field=field, precision=precision,angle_precision = precision-2);
        }
                
    }

}

// struct PlanarNorms {
//     num_faces: usize,
//     faces:  Vec<UnitNorms>,
// }


// impl PlanarNorms {

//     fn new(&um: UnitNorms) -> PlanarNorms {


//         let mut face_basis_defined: Vec<bool>;
//         let mut num_faces: usize = 0;

//         for (idx1, idx2, delta_vector) in izip!(&x.first_index,&x.second_index,&x.data){
            
        
//         }
        

        
//     }
// }


// struct CoordinateVector {

//     size: usize,
//     data:  Vec<Coordinate>,
// }

// struct CoordinateDifferences {

//     size: usize,
//     data:  Vec<Coordinate>,
//     first_index : Vec <usize>,
//     second_index : Vec <usize>,
//     mean_magnitude : f64,
//     magnitude_range : f64,
//     signs: Vec <f64>,
//     dots : Vec <f64>,
// }




fn main() {
    
    // Triangle, d3: 3 vertices, 1 face
    // Tetrahedron, d4: 4 vertices, 4 faces
    // Triangular bipyramid, d6 (not a cube): 5 vertices, 6 faces (at least with these initial values. This is not the only stable state.)
    // Equilateral octohedron, Square bipyramid, d8: 6 vertices (at each of the three axes)
    // Pentagonal bipyramid, d10: 7 vertices, 10 faces
    // Cube, d6: 8 vertices, 6 faces


    // Dodecahedron: d12 20 vertices, 12 faces
    // Icosahedron: d20 12 vertices, 20 faces
    let args: Vec<String> = env::args().collect();

    if args.len() <= 1 {
        println!("One argument is needed to specify the number of vertices which must be larger or equal to 3.");
        exit(1);
    }
    let number_of_vertices: usize = match args[1].parse() {
        Ok(num) => { 
            if num < 3 {
                println!("The number of vertices must be larger than or equal to 3.");
                exit(1);
            } else {
                num
            }
        },
        Err(e) => {
            println!("The first argument must be the number of vertices and must be larger than or equal to 3.");
            exit(1)
        }
    };
    

    let now = Instant::now();
    const SCALE : f64 =  0.1;
    const STOP_POWER : i32 = 10;
    let stop = 10_f64.powi(-STOP_POWER);
    const PRECISION: usize = STOP_POWER as usize + 1; 
    const USE_RANDOM_VERTICES : bool = false;
    let mut coordinates : CoordinateVector;


    // let data = vec![x1,x2,x3,x4,x5,x6];
    if USE_RANDOM_VERTICES {
        coordinates = CoordinateVector::new_from_random_vertices(number_of_vertices);
    } else {
        coordinates = CoordinateVector::new_from_fixed_sequence(number_of_vertices);
    }
    
    
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
        let (new_coordinates, new_max_dx) = coordinates.reposition(&coordinate_differences, scale, counter, NUMBER_OF_CYCLES_BETWEEN_PRINT);
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
            print!("max_dx = {:.precision$}, new_max_dx = {:.precision$}, %diff = {:.precision$}\n",
                    max_dx, new_max_dx, 1. - new_max_dx/max_dx, precision = PRECISION+2);
            print_timer += now.elapsed().as_secs_f64() - print_sub_timer;
        }
        if new_max_dx < stop {
            let print_sub_timer = now.elapsed().as_secs_f64();
            let elapsed_time_excluding_output = now.elapsed().as_secs_f64();
            println!("Final values dx stop **************************");
            println!("loop: counter = {} **************************",counter);
            coordinates.print(PRECISION);
            coordinate_differences.print(PRECISION);
            print!("max_dx = {:.precision$}, new_max_dx = {:.precision$}, %diff = {:.precision$}\n",
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
            print!("max_dx = {:.precision$}, new_max_dx = {:.precision$}, %diff = {:.precision$}\n",
                            max_dx, new_max_dx, 1. - new_max_dx/max_dx, precision = PRECISION+2);
            println!("Elpased time ={:0.3} ms", 1000.0 * now.elapsed().as_secs_f64());
            print_timer += now.elapsed().as_secs_f64() - print_sub_timer;
            break;
        }
    }
    let edge_dots_and_unit_norms = CoordinateDifferences::get_edge_dots_and_unit_norms(&coordinate_differences.first_index,&coordinate_differences.second_index,&coordinate_differences.data);
    let unit_norms: UnitNorms = UnitNorms::new(&edge_dots_and_unit_norms);
    unit_norms.print(PRECISION);
    println!("*********************************************************************************************************************************");
    let unit_norms = unit_norms.sort_by_face();
    unit_norms.print(PRECISION);

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
