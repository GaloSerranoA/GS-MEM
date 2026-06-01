//! GS-MEM HTTP server.

#[tokio::main]
async fn main() -> Result<(), gs_mem_server::AppError> {
    gs_mem_server::run().await
}
