use itertools::izip;

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

    // fn delta_and_mag(&self, pt:&Coordinate) -> (Coordinate, f64) {

    //     (pt.delta(&self), pt.delta(&self).mag())
    // }
    
    // Vector from self to pt
    fn delta(&self, pt:&Coordinate) -> Coordinate {

        pt.sub(self)
    }

    fn calc_mag(x:f64,y:f64,z:f64) -> f64 {

        (x.powi(2)  + y.powi(2) + z.powi(2)).sqrt()
    }

    // fn data_and_mag(&self) -> (Coordinate,f64) {

    //     return(*self,self.mag())
    // }

    // fn distance(&self, pt:&Coordinate) -> f64 {

    //     self.delta(&pt).mag()
        
    // }

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

    fn reposition(&self, pt: &Coordinate, scale:f64) -> Coordinate {

        // println!("reposition");
        // let x = &self.delta(pt);
        // println!("x = ");
        // x.print();
        // let dx = x.mult(scale);
        // println!("dx = ");
        // dx.print();
        // println!("self = ");
        // self.print();
        // let y = self.add(&dx);
        // println!("y = ");
        // y.print();
        // let z = y.make_unit_vector();
        // println!("z = ");
        // z.print();

        // let w = self.add(&self.delta(pt).mult(scale)).make_unit_vector();
        // println!("w = ");
        // w.print();


        self.add(&self.delta(pt).mult(scale)).make_unit_vector()
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

        self.delta(&pt).mag < eps

    }

    fn print(&self) {
    
        println!("( {:.3} , {:.3} , {:.3} ), |x| = {:.3}",self.x,self.y,self.z, self.mag)
    }



}


struct CoordinateVector {

    size: usize,
    data:  Vec<Coordinate>,
    first_index : Option <Vec <usize>>,
    second_index : Option <Vec <usize>>,
}

impl CoordinateVector {
    
    // fn data_and_mag(data : Vec<Coordinate> ) -> (Vec<Coordinate>,Vec<f64>,usize){

    //     let mut newdata : &Vec <Coordinate> = &Vec::new() ;
    //     for coordinate in data[0..data.len()].iter(){
    //         newdata.push(&oordinate);
    //     }
    //     return (newdata, magnitudes,data.len())

    // }
    
    fn new(data : Vec<Coordinate> ) -> CoordinateVector{
        // self.data = data;
        // self.size = self.data.len();
        // self

        // let (newdata, magnitudes, len) = CoordinateVector::data_and_mag(data);

        CoordinateVector{

            size: data.len(),
            data : data, // note this should move data
            first_index : None,
            second_index : None,
        }
    }

    fn print(&self) {

        match &self.first_index{
            Some(vec_idx1) => match &self.second_index{
                Some(vec_idx2) => {
                    println!("Length = {}",self.size);
                    for (idx1, idx2, coordinate) in izip!(vec_idx1, vec_idx2, &self.data) {
                        print!("({}, {}): ",idx1,idx2);
                        coordinate.print();
                    }
                },
                None =>  (), // only need to output the None case once
            },
            None => {
                println!("Length = {}",self.size);
                for coordinate in &self.data {
                    coordinate.print();
                }
            },
        }
    }

    fn construct_differences(&self) -> CoordinateVector {
        
        let mut data : Vec<Coordinate> = Vec::new();
        let mut first_idx : Vec <usize>  = Vec::new();
        let mut second_idx : Vec <usize> = Vec::new();
        let mut magnitudes : Vec <f64> = Vec::new();
        for (idx1, coordinate_1) in self.data[0..self.size-1].iter().enumerate() {
            for (idx2, coordinate_2) in self.data[idx1+1..self.size].iter().enumerate(){
                first_idx.push(idx1);
                second_idx.push(idx1+idx2+1);
                let difference = coordinate_2.delta(coordinate_1);
                data.push(difference);
            }
        }
        CoordinateVector{

            size: data.len(),
            data : data,
            first_index : Some(first_idx),
            second_index : Some(second_idx),
        }      

    }


}


fn main() {

    let x1 = Coordinate::new(0. , 0., 1.);
    let mut x2 = Coordinate::new(1. , 0.,-0.);
    let mut x3 = Coordinate::new(0. , 1.,-0.);
    let mut x4 = Coordinate::new(0. , 1.0/ 2.0_f64.powf(0.5), -1.0/2.0_f64.powf(0.5));

    let data = vec![x1,x2,x3,x4];
    let mut coordinates = CoordinateVector::new(data);
    // let mut max_error = 10000.0_f64;
    // let mut max_magnitude = 0.0_f64;
    // let mut min_magnitude = 0.0_f64;
    // let mut max_idx1 = 0_usize;
    // let mut max_idx2 = 0_usize;
    // let mut this_magnitude = 0.0_f64;
    // let mut max_distance = Coordinate::new(0. , 0., 0.);
    // let mut difference= Coordinate::new(0. , 0., 0.);
    // let proportional_change = 1e-2_f64;
    // let mut iteration_number = 0_usize;

    coordinates.print();
    let coordinate_differences = coordinates.construct_differences();
    coordinate_differences.print()




    // while max_error > 0.01 {
    //     max_magnitude = 0.0;
    //     max_idx1 = 0;
    //     max_idx2 = 1;

    //     for (idx1, coordinate_1) in coordinates.data[0..coordinates.size-1].iter().enumerate() {
    //         // println!("idx1 = {}",idx1);
    //         for (idx2, coordinate_2) in coordinates.data[idx1+1..coordinates.size].iter().enumerate(){
    //             // println!("idx2 = {}",idx2 + idx1 + 1);
    //             // let cd2 =  coordinate_2;
    //             // let cd1 =  coordinate_1;

    //             // difference = cd2.delta(cd1); 
    //             // Note difference is an &Coordinate!!!
    //             // print!("({},{}): ",idx1, idx2 + idx1 + 1);
                
    //             let (difference, this_magnitude) = coordinate_2.delta_and_mag(coordinate_1);
    //             difference.print();
    //             if this_magnitude > max_magnitude{
    //                 max_magnitude = this_magnitude;
    //                 max_idx1 = idx1;
    //                 max_idx2 = idx2;
    //                 max_distance = difference;
    //             }
    //             if this_magnitude < min_magnitude{
    //                 min_magnitude = this_magnitude;
    //             }
    //             max_error = max_magnitude - min_magnitude
    //         }
    //     }
    //     println!("Iteration {}",iteration_number);
    //     print!("({},{}): ",max_idx1,max_idx2+max_idx1+1);
    //     max_distance.print();
    //     let idx2 = max_idx2+max_idx1+1;
    //     let idx1 = max_idx1;
    //     coordinates.data[idx2] = coordinates.data[idx2].reposition(&coordinates.data[idx1], proportional_change);
    //     coordinates.print();
    //     iteration_number += 1;


    // }

}
