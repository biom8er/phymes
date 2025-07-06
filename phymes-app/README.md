# PHYMES: Parallel HYpergraph MEssaging Streams

App crate

<!--- ANCHOR: synopsis --->

## Synopsis

The PHYMES application crate implements the UI/UX for querying the `SessionContext` and publishing on and subscribing to subjects for predefined `SessionPlans` directly or through a chat interface using [dioxus](https://dioxuslabs.com/). Application builds for Web, Desktop (all OSs), and mobile (all OSs) are theoretically supported, but only Web and Desktop are optimized for currently. The application takes a full-stack approach whereby the frontend is handled by this crate using dioxus and the backend is handled by the phymes-server crate using tokio. 

<!--- ANCHOR_END: synopsis --->