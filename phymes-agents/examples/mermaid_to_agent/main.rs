mod run_main;
use run_main::run_main;

fn main() {
    match run_main() {
        Ok((flowchart, erdiagram)) => {
            println!("Flowchart: {flowchart}");
            println!("ER Diagram: {erdiagram}");
        }
        Err(err) => {
            println!("Build errors: {err:?}");
        }
    }
}