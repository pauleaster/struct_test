use itertools::izip;
// #[macro_use] extern crate text_io; // used to pause 
// use rand::prelude::*;
use rand::distributions::{Distribution, Uniform};
use std::f64::consts::PI;


const MAX_NUM : f64 = 1e6;
const MIN_NUM : f64 = 0.0;
const SCALE : f64 =  0.01;
const STOP_POWER : i32 = 10;


struct Coordinate {

    x: f64,
    y: f64,
    z: f64,
    mag: f64,
}



impl Coordinate {


    fn new(x : f64, y: f64, z: f64 ) -> Coordinate{

        Coordinate {
            x: x, 
            y: y,
            z: z,
            mag : Coordinate::calc_mag(x,y,z),
        }
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
        let theta_range = Uniform::from(0.0..2.0 * PI);
        let phi_range = Uniform::from(0.0..PI);

        let theta = theta_range.sample(&mut rng);
        let phi = phi_range.sample(&mut rng);

        let x : f64 = theta.cos() * phi.sin();
        let y : f64 = theta.sin() * phi.sin();
        let z : f64 = phi.cos();

        let mag: f64 = 1.0;

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
            x: x / mag,
            y: y,
            z: z / mag ,
            mag : 1.0,
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
            x: x,
            y: y,
            z: z ,
            mag : Coordinate::calc_mag(x,y,z),
        }
        
    }

    fn sub(&self, pt:&Coordinate) -> Coordinate {

        let x:f64 = self.x - pt.x;
        let y:f64 = self.y - pt.y;
        let z:f64 = self.z - pt.z;

        Coordinate {
            x: x,
            y: y,
            z: z ,
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
            x: x,
            y: y,
            z: z ,
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

        self.sub(&pt).mag < eps

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

    fn dot (&self, pt:&Coordinate) -> f64 {

        self.x * pt.x + self.y * pt.y + self.z * pt.z
    }


}


struct CoordinateVector {

    size: usize,
    data:  Vec<Coordinate>,
}

impl CoordinateVector {
    
    
    fn new(data : Vec<Coordinate> ) -> CoordinateVector{


        CoordinateVector{

            size: data.len(),
            data : data, // note this should move data
        }
    }


    fn clone(&self) -> CoordinateVector {

        let mut data:  Vec<Coordinate> = Vec::new();



        for coordinate in &self.data {
            data.push(coordinate.clone())
        }
        return CoordinateVector::new(data);
    }

    
    fn zero(&self) -> CoordinateVector{

        let mut zero_vectors = self.clone();

        for index in 0..self.size {
            zero_vectors.data[index] = Coordinate::zero();
        }
        zero_vectors
    }


    fn print(&self, precision: usize) {

        let field:usize = precision + 4; 

        println!("Length = {:3}",self.size);
        for coordinate in &self.data {
            coordinate.print('\n', precision);
        }
    }


    fn reposition (&self, differences: &CoordinateDifferences, scale:f64) -> CoordinateVector {

        
        let mut result = self.clone();
        let mut forces = self.zero();
        for (idx1, idx2, delta_vector) in izip!(&differences.first_index,&differences.second_index,&differences.data){
            if *idx1 != 0 {
                forces.data[*idx1]  = forces.data[*idx1].add(&delta_vector.mult(delta_vector.mag.powi(-3) * scale)); // du_hat/|u|^2
                
            }
            if *idx2 != 0 {
                forces.data[*idx2]  = forces.data[*idx2].add(&delta_vector.mult(-1.0 * delta_vector.mag.powi(-3) * scale));
            }
        }
        for (idx, force) in forces.data.iter().enumerate() {
            if idx == 0 {
                continue;
            }
            result.data[idx] = result.data[idx].add(force).make_unit_vector();
        }
        return result;
    }
}

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

        
        let mut data : Vec<Coordinate> = Vec::new();
        let mut first_idx : Vec <usize>  = Vec::new();
        let mut second_idx : Vec <usize> = Vec::new();
        let mut mags : Vec <f64> = Vec::new();
        let mut dot_products : Vec <f64> = Vec::new();
        let mut mag_sum = 0_f64;
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
        let mean_magnitude = mag_sum  / (data.len() as f64);
        let delta_mags : Vec <f64> = mags.iter().map(|&x| x - mean_magnitude).collect(); //::<Vec<f64>>();
        let signs : Vec <f64> = mags.iter().map(|&x| (x - mean_magnitude)/(x - mean_magnitude).abs() as f64 * 2.0 - 1.0 ).collect(); 
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


        CoordinateDifferences{

            size: data.len(),
            data : data,
            first_index : first_idx,
            second_index : second_idx,
            mean_magnitude : mean_magnitude,
            magnitude_range : mag_range,
            signs : signs,
            dots : dot_products,
        }

    }

    fn print(&self, precision: usize) {

        let field:usize = precision + 4; 

        println!("Length = {:3}",self.size);
        println!("E(|x|) = {:field$.precision$}",&self.mean_magnitude, precision=precision, field=field);
        for (idx1, idx2, coordinate, dot, sign) in izip!(&self.first_index, &self.second_index, &self.data, &self.dots, &self.signs) {
            print!("({:3}, {:3}): ",idx1,idx2);
            coordinate.print(',',precision);
            print!(" |x| - E(|x|) = {:field$.precision$}, Sign = [{:+.0}]",
                                                        coordinate.mag - &self.mean_magnitude, sign,
                                                        precision=precision, field=field);
            println!(" <a.b> = {:field$.precision$}",dot, field=field, precision=precision);
        }
        println!("max(|x|) - min(|x|) = {:field$.precision$}",&self.magnitude_range, precision=precision, field=field);
    }

}


fn main() {


    let x1 = Coordinate::unit(0. , 0., 1.);
    let mut x2 = Coordinate::new_random();
    x2 = x2.project_onto_xz();
    let x3 = Coordinate::new_random();
    let x4 = Coordinate::new_random();
    let x5 = Coordinate::new_random();
    let x6 = Coordinate::new_random();

    let stop = 10_f64.powi(-STOP_POWER);
    const PRECISION: usize = STOP_POWER as usize + 1; 

    let data = vec![x1,x2,x3,x4,x5,x6];
    let mut coordinates = CoordinateVector::new(data);
    // let mut new_coordinates = CoordinateVector::new(data);
    let mut prev_scale : f64;
    const NUMBER_OF_CYCLES_BETWEEN_PRINT: usize = 1;
    

    let mut scale = SCALE;
    let mut max_difference:f64;
    let mut new_max_difference:f64;
    let mut counter: usize = 0;
    // let mut prev_max_distance: f64 = 1000.0;
    println!("Initial values **************************");
    coordinates.print(PRECISION);
    let mut coordinate_differences = CoordinateDifferences::new(&coordinates);
    coordinate_differences.print(PRECISION);
    max_difference = coordinate_differences.magnitude_range;
    loop {
        println!("loop: counter = {} **************************",counter);
        let new_coordinates = coordinates.reposition(&coordinate_differences, scale);
        if counter % NUMBER_OF_CYCLES_BETWEEN_PRINT == 0 {
            new_coordinates.print(PRECISION);
        }
        let new_coordinate_differences = CoordinateDifferences::new(&new_coordinates);
        if counter % NUMBER_OF_CYCLES_BETWEEN_PRINT == 0 {
            new_coordinate_differences.print(PRECISION);
        }
        new_max_difference = new_coordinate_differences.magnitude_range;

        if counter % NUMBER_OF_CYCLES_BETWEEN_PRINT == 0 {
            print!("max_difference = {:.precision$}, new_max_difference = {:.precision$}, %diff = {:.precision$}\n",max_difference, new_max_difference, 1. - new_max_difference/max_difference, precision = PRECISION+2);
        }
        if max_difference < stop {
            break;
        }
        prev_scale = scale;
        scale =  scale.min(max_difference * 0.01);
        if counter % NUMBER_OF_CYCLES_BETWEEN_PRINT == 0 {
            print!("prev_scale = {:.precision$}, scale = {:.precision$}, %diff = {:.precision$}\n",prev_scale, scale, 1. - scale/prev_scale, precision = PRECISION+2);
        }
 
        coordinates = new_coordinates;
        coordinate_differences = new_coordinate_differences;
        max_difference = new_max_difference;
        counter += 1;
        if counter > 10_usize.pow(6) {
            break;
        }
    }
    // print!("\n\n{:?}",all_scales);
}
