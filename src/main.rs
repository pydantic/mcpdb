use clap::Parser;
use fastembed::TextEmbedding;
use rmcp::transport::sse_server::SseServer;
use rmcp::{
    ServerHandler,
    handler::server::wrapper::Json,
    model::{ServerCapabilities, ServerInfo},
    schemars, tool,
    transport::sse_server::SseServerConfig,
};
use serde_json::json;
use sqlx::{Executor, Pool, Postgres};
use std::net::{IpAddr, SocketAddr};
use std::sync::LazyLock;

#[derive(Debug, Clone)]
pub struct MemoryServer {
    pool: Pool<Postgres>,
}

#[tool(tool_box)]
impl MemoryServer {
    #[tool(description = "Generate embeddings for a given text using an LLM")]
    async fn generate_embeddings(
        &self,
        #[tool(param)]
        #[schemars(description = "The input text to generate embeddings for")]
        text: String,
    ) -> Json<Vec<f32>> {
        Json(generate_embedding(text).await)
    }

    #[tool(description = "Store a value in memory")]
    async fn store(
        &self,
        #[tool(param)]
        #[schemars(description = "The key to store the value under")]
        key: Option<String>,
        #[tool(param)]
        #[schemars(description = "The value to store, as a JSON object")]
        value: serde_json::Value,
        // TODO add on_conflict option
    ) -> Json<serde_json::Value> {
        let key = key.unwrap_or_else(|| "default".to_string());
        let embedding = generate_embedding(value.to_string()).await;

        let mut client = self.pool.acquire().await.unwrap();
        sqlx::query("INSERT INTO memory (key, value, embedding) VALUES ($1, $2, $3) ON CONFLICT (key) DO UPDATE SET value = $2, embedding = $3")
            .bind(&key)
            .bind(value)
            .bind(&embedding)
            .execute(&mut *client)
            .await
            .expect("Failed to execute query");

        Json(json!({
            "key": key,
            // TODO: maybe return the result of the insert (inserted / replaced)
        }))
    }

    #[tool(description = "Retrieve a value from memory")]
    async fn retrieve(
        &self,
        #[tool(param)]
        #[schemars(description = "The key to retrieve the value for")]
        key: String,
    ) -> Json<serde_json::Value> {
        let mut client = self.pool.acquire().await.unwrap();
        let row: (String, serde_json::Value) =
            sqlx::query_as("SELECT key, value FROM memory WHERE key = $1")
                .bind(&key)
                .fetch_one(&mut *client)
                .await
                .expect("Failed to execute query");

        Json(json!({
            "key": row.0,
            "value": row.1,
        }))
    }

    #[tool(description = "Delete a value from memory")]
    async fn delete(
        &self,
        #[tool(param)]
        #[schemars(description = "The key to delete the value for")]
        key: String,
    ) -> Json<serde_json::Value> {
        let mut client = self.pool.acquire().await.unwrap();
        let row: (String, serde_json::Value) =
            sqlx::query_as("DELETE FROM memory WHERE key = $1 RETURNING key, value")
                .bind(&key)
                .fetch_one(&mut *client)
                .await
                .expect("Failed to execute query");

        Json(json!({
            "key": row.0,
            "value": row.1,
        }))
    }

    #[tool(description = "Search memory for similar values")]
    async fn search(
        &self,
        #[tool(param)]
        #[schemars(description = "The query text to search for")]
        query: String,
        #[tool(param)]
        #[schemars(description = "The number of results to return")]
        limit: Option<usize>,
    ) -> Json<Vec<serde_json::Value>> {
        let mut client = self.pool.acquire().await.unwrap();
        let limit = limit.unwrap_or(10);
        let embedding = generate_embedding(query).await;

        let rows: Vec<(String, serde_json::Value)> = sqlx::query_as(
            "SELECT key, value FROM memory ORDER BY (embedding <-> $1::vector) LIMIT $2",
        )
        .bind(&embedding)
        .bind(limit as i64)
        .fetch_all(&mut *client)
        .await
        .expect("Failed to execute query");

        Json(
            rows.into_iter()
                .map(|row| {
                    json!({
                        "key": row.0,
                        "value": row.1,
                    })
                })
                .collect(),
        )
    }
}

#[tool(tool_box)]
impl ServerHandler for MemoryServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some("A simple \"memory\" for agents to store data".into()),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}

async fn generate_embedding(text: String) -> Vec<f32> {
    // FIXME error handling

    static EMBEDDER: LazyLock<TextEmbedding> = LazyLock::new(|| {
        TextEmbedding::try_new(Default::default()).expect("Failed to create TextEmbedding")
    });

    // TODO might want to use a rayon thread pool for this
    let mut embeddings = tokio::task::spawn_blocking(move || EMBEDDER.embed(vec![text], None))
        .await
        .expect("failed to join")
        .expect("failed to generate embeddings");
    embeddings.pop().expect("should be one embedding")
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Host address to bind the server to
    #[arg(long, default_value = "127.0.0.1")]
    host: IpAddr,

    /// Port to bind the server to
    #[arg(long, default_value = "3456")]
    port: u16,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // TODO add proper logfire configuration (optional)
    let logfire = logfire::configure()
        .send_to_logfire(false)
        .install_panic_handler()
        .finish()?;

    let pool = sqlx::Pool::<sqlx::Postgres>::connect("postgres://postgres:postgres@localhost:5432")
        .await
        .expect("Failed to create pool");

    let mut client = pool.acquire().await.expect("Failed to acquire client");

    // TODO: could use proper migrations here

    client
        .execute("CREATE EXTENSION IF NOT exists vector;")
        .await?;

    client
        .execute(
            "CREATE TABLE IF NOT EXISTS memory (
        id SERIAL PRIMARY KEY,
        key TEXT NOT NULL,
        value JSONB NOT NULL,
        embedding vector NOT NULL,
        UNIQUE (key)
    )",
        )
        .await?;

    // TODO default SseServer::serve seems to set inconsistent paths, which seems
    // to be against MCP spec. Fix upstream?
    let ct = SseServer::serve_with_config(SseServerConfig {
        bind: SocketAddr::new(cli.host, cli.port),
        sse_path: "/sse".into(),
        post_path: "/sse".into(),
        ct: Default::default(),
        sse_keep_alive: None, // foo
    })
    .await?
    .with_service(move || MemoryServer { pool: pool.clone() });

    tokio::signal::ctrl_c().await?;
    ct.cancel();
    logfire.shutdown()?;
    Ok(())
}
