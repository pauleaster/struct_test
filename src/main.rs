struct Coordinate {

    x: f64,
    y: f64,
    z: f64,
}

impl Coordinate {

    fn new(x : f64, y: f64, z: f64 ) -> Coordinate{
        Coordinate {
            x: 0.0, 
            y: 0.0,
            z: 0.0
        }
    }

    fn add(self, pt:Coordinate) -> Coordinate {

        Coordinate {
            x: self.x + pt.x ,
            y: self.y + pt.y ,
            z: self.z + pt.z 
        }
        
    }

}


fn main() {
    let mut c = Coordinate::new(None,None);
    let a = Coordinate { x:245, y:216 };
    let b = Coordinate { x:6, y:7 };
    c.x = 4;
    c.y = 33;
    let d = a.add(b).add(c);
    println!("{} {}",d.x,d.y)

}
