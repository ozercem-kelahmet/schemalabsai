from flask import Flask, request, jsonify
from pyspark.sql import SparkSession
import os, uuid, logging, threading

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

SPARK_MASTER = os.getenv("SPARK_MASTER", "local[4]")
SPARK_MEMORY = os.getenv("SPARK_MEMORY", "4g")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/app/uploads")

# Spark session - lazy init, bir kez oluştur
_spark = None
_spark_lock = threading.Lock()

def get_spark():
    global _spark
    if _spark is None:
        with _spark_lock:
            if _spark is None:
                _spark = SparkSession.builder \
                    .appName("SchemaLabs") \
                    .master(SPARK_MASTER) \
                    .config("spark.executor.memory", SPARK_MEMORY) \
                    .config("spark.driver.memory", SPARK_MEMORY) \
                    .config("spark.sql.shuffle.partitions", "8") \
                    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
                    .getOrCreate()
                _spark.sparkContext.setLogLevel("WARN")
                log.info(f"Spark initialized: master={SPARK_MASTER}")
    return _spark

# Job store
jobs = {}

@app.route("/health")
def health():
    return jsonify({"status": "ok", "spark_master": SPARK_MASTER})

@app.route("/api/v1/jobs", methods=["POST"])
def submit_job():
    req = request.json
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {"status": "running", "job_id": job_id}
    
    # Background thread'de çalıştır
    t = threading.Thread(target=run_job, args=(job_id, req))
    t.daemon = True
    t.start()
    
    return jsonify({"job_id": job_id, "status": "running"})

@app.route("/api/v1/jobs/<job_id>")
def get_job(job_id):
    job = jobs.get(job_id, {"status": "not_found"})
    return jsonify(job)

def run_job(job_id, req):
    try:
        job_type = req.get("job_type", "")
        conn_type = req.get("conn_type", "")
        output_path = req.get("output_path", f"{UPLOAD_DIR}/{job_id}_output.csv")
        config = req.get("config", {})

        spark = get_spark()
        df = None

        # SQL tabanlı (PostgreSQL, Supabase, MySQL)
        if job_type == "export_sql":
            jdbc_url = config.get("jdbc_url")
            table = config.get("table")
            driver = config.get("driver", "org.postgresql.Driver")
            
            df = spark.read \
                .format("jdbc") \
                .option("url", jdbc_url) \
                .option("dbtable", table) \
                .option("driver", driver) \
                .option("fetchsize", "10000") \
                .option("numPartitions", "4") \
                .load()

        # Snowflake
        elif job_type == "export_snowflake":
            df = spark.read \
                .format("net.snowflake.spark.snowflake") \
                .option("sfURL", config.get("url")) \
                .option("sfUser", config.get("user")) \
                .option("sfPassword", config.get("password")) \
                .option("sfDatabase", config.get("database")) \
                .option("sfWarehouse", config.get("warehouse")) \
                .option("dbtable", config.get("table")) \
                .load()

        # MongoDB
        elif job_type == "export_mongodb":
            df = spark.read \
                .format("mongo") \
                .option("uri", config.get("uri")) \
                .option("database", config.get("database")) \
                .option("collection", config.get("collection")) \
                .load()

        # Databricks
        elif job_type == "export_databricks":
            jdbc_url = config.get("jdbc_url")
            table = config.get("table")
            df = spark.read \
                .format("jdbc") \
                .option("url", jdbc_url) \
                .option("dbtable", table) \
                .option("fetchsize", "10000") \
                .load()

        # CSV parse (büyük upload)
        elif job_type == "parse_csv":
            input_path = config.get("input_path")
            df = spark.read \
                .option("header", "true") \
                .option("inferSchema", "true") \
                .option("multiLine", "true") \
                .option("escape", '"') \
                .csv(input_path)

        # CSV merge job
        elif job_type == "merge_csv":
            input_paths = config.get("input_paths", "").split(",")
            input_paths = [p.strip() for p in input_paths if p.strip()]
            row_count = merge_csvs_with_spark(spark, input_paths, output_path)
            jobs[job_id] = {
                "status": "completed",
                "job_id": job_id,
                "output_path": output_path,
                "row_count": row_count
            }
            return

        # Preprocess job
        elif job_type == "preprocess":
            input_paths = config.get("input_paths", config.get("input_path", ""))
            target_col = config.get("target_col", None)
            paths = [p.strip() for p in input_paths.split(",") if p.strip()]
            dfs = []
            for p in paths:
                try:
                    df_temp = spark.read                         .option("header", "true")                         .option("inferSchema", "true")                         .option("multiLine", "true")                         .option("escape", chr(34))                         .csv(p)
                    dfs.append(df_temp)
                    log.info(f"[PREPROCESS] Loaded {p}: {df_temp.count()} rows")
                except Exception as e:
                    log.error(f"[PREPROCESS] Failed to load {p}: {e}")
            if not dfs:
                jobs[job_id] = {"status": "failed", "error": "No files loaded"}
                return
            if len(dfs) > 1:
                from functools import reduce
                df = reduce(lambda a, b: a.unionByName(b, allowMissingColumns=True), dfs)
            else:
                df = dfs[0]
            df = preprocess_dataframe(df, target_col)

        if df is None:
            jobs[job_id] = {"status": "failed", "error": f"Unknown job_type: {job_type}"}
            return

        row_count = df.count()
        
        # CSV olarak yaz - tek dosya
        df.coalesce(1).write \
            .mode("overwrite") \
            .option("header", "true") \
            .csv(output_path + "_tmp")
        
        # Spark parça dosyaları birleştir
        import glob, shutil
        parts = glob.glob(output_path + "_tmp/part-*.csv")
        if parts:
            shutil.copy(parts[0], output_path)
            shutil.rmtree(output_path + "_tmp")

        log.info(f"Job {job_id} completed: {row_count} rows → {output_path}")
        jobs[job_id] = {
            "status": "completed",
            "job_id": job_id,
            "output_path": output_path,
            "row_count": row_count
        }

    except Exception as e:
        log.error(f"Job {job_id} failed: {e}")
        jobs[job_id] = {"status": "failed", "job_id": job_id, "error": str(e)}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)

# Merge job - birden fazla CSV'yi Spark ile merge et
def merge_csvs_with_spark(spark, input_paths, output_path):
    dfs = []
    for path in input_paths:
        try:
            df = spark.read \
                .option("header", "true") \
                .option("inferSchema", "true") \
                .option("multiLine", "true") \
                .option("escape", '"') \
                .csv(path)
            dfs.append(df)
            log.info(f"Loaded {path}: {df.count()} rows")
        except Exception as e:
            log.error(f"Failed to load {path}: {e}")
    
    if not dfs:
        return 0
    
    # UNION ALL ile merge
    from functools import reduce
    merged = reduce(lambda a, b: a.unionByName(b, allowMissingColumns=True), dfs)
    row_count = merged.count()
    
    merged.coalesce(1).write \
        .mode("overwrite") \
        .option("header", "true") \
        .csv(output_path + "_tmp")
    
    import glob, shutil
    parts = glob.glob(output_path + "_tmp/part-*.csv")
    if parts:
        shutil.copy(parts[0], output_path)
        shutil.rmtree(output_path + "_tmp")
    
    log.info(f"Merged {len(dfs)} files: {row_count} rows → {output_path}")
    return row_count

def preprocess_dataframe(df, target_col=None):
    from pyspark.sql import functions as F
    from pyspark.sql.types import NumericType, StringType
    for field in df.schema.fields:
        if isinstance(field.dataType, NumericType):
            df = df.fillna({field.name: 0})
        else:
            df = df.fillna({field.name: ""})
    if target_col and target_col in df.columns:
        class_counts = df.groupBy(target_col).count()
        max_count = class_counts.agg(F.max("count")).collect()[0][0]
        min_count = class_counts.agg(F.min("count")).collect()[0][0]
        if max_count / max(min_count, 1) > 5:
            log.info(f"[PREPROCESS] Class imbalance detected, applying oversampling")
            dfs = []
            counts = {row[target_col]: row["count"] for row in class_counts.collect()}
            for cls, cnt in counts.items():
                cls_df = df.filter(F.col(target_col) == cls)
                ratio = int(max_count / cnt)
                if ratio > 1:
                    cls_df = cls_df.sample(withReplacement=True, fraction=float(ratio))
                dfs.append(cls_df)
            from functools import reduce
            df = reduce(lambda a, b: a.union(b), dfs)
    return df
