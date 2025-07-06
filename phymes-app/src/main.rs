// Dioxus imports
use dioxus::prelude::*;

// UI components
mod ui;
use ui::main_window::main_window;

// CSS
static MAIN_CSS: Asset = asset!("/assets/main.css");

fn main() {
    dioxus::launch(app);
}

fn app() -> Element {
    // render the UI
    rsx! {
        document::Link { rel: "stylesheet", href: MAIN_CSS }
        div {
            id: "container",
            main_window {}
        }
    }
}
