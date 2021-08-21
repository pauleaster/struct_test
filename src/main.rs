struct Coordinate {

    x: f64,
    y: f64,
    z: f64,
}

impl Coordinate {


    fn new(x : f64, y: f64, z: f64 ) -> Coordinate{
        Coordinate {
            x: x, 
            y: y,
            z: z,
        }
    }

    fn add(self, pt:Coordinate) -> Coordinate {

        Coordinate {
            x: self.x + pt.x ,
            y: self.y + pt.y ,
            z: self.z + pt.z 
        }
        
    }

    fn sub(self, pt:Coordinate) -> Coordinate {

        Coordinate {
            x: self.x - pt.x ,
            y: self.y - pt.y ,
            z: self.z - pt.z 
        }
        
    }

    fn delta(self, pt:Coordinate) -> Coordinate {

        pt.sub(self)
    }

    fn mag(self) -> f64 {

        (self.x.powf(2.)  + self.y.powf(2.) + self.z.powf(2.)).sqrt()
    }

    fn distance(self, pt:Coordinate) -> f64 {

        self.delta(pt).mag()
        
    }

    fn first_quadrant(self) -> Coordinate {

        Coordinate {
            x: self.x.abs() ,
            y: self.y.abs() ,
            z: self.z.abs() 
        }
        
    }

}


struct CoordinateVector {

    size: usize,
    data:  Vec<Coordinate>,
}

impl CoordinateVector {
    
    fn new(data : Vec<Coordinate> ) -> CoordinateVector{
        // self.data = data;
        // self.size = self.data.len();
        // self

        CoordinateVector{

            size: data.len(),
            data : data,
        }
    }

    fn print(self) {

        println!("Length = {}",self.size);
        for coordinate in self.data {
            println!("( {:.3} , {:.3} , {:.3} )",coordinate.x,coordinate.y,coordinate.z)
        }

    }

    fn get_size(self) -> usize {
        self.size
    }


}


fn main() {
    let x1 = Coordinate::new(0. , 0.,-1.);
    let mut x2 = Coordinate::new(1. , 0.,-0.);
    let mut x3 = Coordinate::new(0. , 1.,-0.);
    let mut x4 = Coordinate::new(0. , 0., -1.);

    let data = vec![x1,x2,x3,x4];
    let mut coordinates = CoordinateVector::new(data);
    let mut max_error = 10000.0;

    // coordinates.print();

    // let size = coordinates.get_size();

    while max_error > 0.01 {
        for vec1 in coordinates.data.into_iter(){

        }
    }



}
