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


fn main() {
    let x1 = Coordinate::new(0. , 0.,-1.);
    let mut x2 = Coordinate::new(1. , 0.,-0.);
    let mut x3 = Coordinate::new(0. , 1.,-0.);
    let mut x4 = Coordinate::new(0. , 0., -1.);

    c.x = 4.;
    c.y = 33.;
    let d = a.add(b).add(c);
    println!("{:.3} {:.3} {:.3}",d.x,d.y,d.z);
    let x:Coordinate = a.add(b);


}
