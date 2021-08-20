struct Coordinate {

    x: u8,
    y: u8,
}

impl Coordinate {

    fn new(x : Option<u8>, y: Option<u8> ) -> Coordinate{
        Coordinate {
            x:x.unwrap_or(0), 
            y:y.unwrap_or(0)
        }
    }

    fn add(self, pt:Coordinate) -> Coordinate {

        Coordinate {
            x: match self.x.checked_add(pt.x) {
                None => panic!("Overflow in x, {} + {} > 255",self.x,pt.x),
                Some(result) => result,
            },
            y: match self.y.checked_add(pt.y) {
                None => panic!("Overflow in y, {} + {} > 255",self.y,pt.y),
                Some(result) => result,
            }
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
