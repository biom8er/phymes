mod run_main;
use run_main::run_main;

fn main() {
    if let Err(err) = run_main() {
        println!("Build errors: {err:?}");
    }
}
