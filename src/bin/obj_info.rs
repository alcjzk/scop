use scop::{Error, Obj, Result};

use std::env;
use std::fs::OpenOptions;

fn main() -> Result<()> {
    let path = env::args()
        .nth(1)
        .ok_or(Error::Generic("obj file path not specified"))?;

    let file = OpenOptions::new().read(true).open(path)?;
    let obj = Obj::from_reader(&file)?;

    println!("{obj:#?}");

    Ok(())
}
