from flask import Flask, request, jsonify
from pyspark.sql import SparkSession
import os, uuid, logging, threading

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

SPARK_MASTER = os.getenv("SPARK_MASTER", "local[4]")
SPARK_MEMORY = os.getenv("SPARK_MEMORY", "4g")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/app/uploads")
GCS_BUCKET = os.getenv("GCS_BUCKET", "schemalabs-prod-us-central1")

def resolve_path(path):
    """GCS key ise indir, local path dön"""
    if not path:
        return path
    if os.path.exists(path):
        return path
    gcs_key = path.replace("gs://", "").replace(f"{GCS_BUCKET}/", "")
    if path.startswith("gs://"):
        gcs_key = path.replace(f"gs://{GCS_BUCKET}/", "")
    if gcs_key.startswith("users/") or gcs_key.startswith("shared/") or gcs_key.startswith("uploads/"):
        try:
            from google.cloud import storage
            client = storage.Client()
            bucket = client.bucket(GCS_BUCKET)
            blob = bucket.blob(gcs_key)
            if blob.exists():
                local_path = f"/tmp/spark_{os.path.basename(gcs_key)}"
                blob.download_to_filename(local_path)
                log.info(f"[GCS] Downloaded: {gcs_key} → {local_path}")
                return local_path
        except Exception as e:
            log.error(f"[GCS] Download failed: {gcs_key} — {e}")
    return path

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
                    .config("spark.jars", "/opt/gcs-connector-hadoop3-shaded.jar") \
                    .config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem") \
                    .config("spark.hadoop.fs.AbstractFileSystem.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS") \
                    .config("spark.hadoop.google.cloud.auth.type", "COMPUTE_ENGINE") \
                    .getOrCreate()
                _spark.sparkContext.setLogLevel("WARN")
                log.info(f"Spark initialized: master={SPARK_MASTER}")
    return _spark


def sanitize_columns(df):
    """Kolon isimlerindeki nokta, virgül, köşeli parantez temizle"""
    for col_name in df.columns:
        clean = col_name.replace(".", "_").replace(",", "_").replace("[", "(").replace("]", ")").replace("{", "(").replace("}", ")")
        if clean != col_name:
            df = df.withColumnRenamed(col_name, clean)
    return df

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
        job_type = req.get("job_type", ""); log.info(f"[JOB] type={job_type} id={job_id}"); log.info(f"[JOB] type={job_type} id={job_id}")
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
            input_path = resolve_path(config.get("input_path"))
            df = spark.read \
                .option("header", "true") \
                .option("inferSchema", "true") \
                .option("multiLine", "true") \
                .option("escape", '"') \
                .csv(input_path)
            df = sanitize_columns(df)

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
            from concurrent.futures import ThreadPoolExecutor
            raw_paths = [p.strip() for p in input_paths.split(",") if p.strip()]
            with ThreadPoolExecutor(max_workers=12) as pool:
                paths = list(pool.map(resolve_path, raw_paths))
            paths = [p for p in paths if p]
            log.info(f"[PREPROCESS] Parallel downloaded {len(paths)} files")
            dfs = []
            for p in paths:
                try:
                    df_temp = spark.read                         .option("header", "true")                         .option("inferSchema", "true")                         .option("multiLine", "true")                         .option("escape", chr(34))                         .csv(p)
                    df_temp = sanitize_columns(df_temp)
                    dfs.append(df_temp)
                    log.info(f"[PREPROCESS] Loaded {p}: {df_temp.count()} rows")
                except Exception as e:
                    log.error(f"[PREPROCESS] Failed to load {p}: {e}")
            if not dfs:
                jobs[job_id] = {"status": "failed", "error": "No files loaded"}
                return
            if len(dfs) > 1:
                from functools import reduce
                import pandas as pd; df = spark.createDataFrame(pd.concat([x.toPandas() for x in dfs], ignore_index=True))
            else:
                df = dfs[0]
            df = preprocess_dataframe(df, target_col)

        if df is None:
            jobs[job_id] = {"status": "failed", "error": f"Unknown job_type: {job_type}"}
            return

        row_count = -1
        
        import pandas as pd, shutil
        local_output = f"/tmp/spark_output_{job_id}.csv"
        df.toPandas().to_csv(local_output, index=False)
        if True:
            pass
            # Upload to GCS
            try:
                from google.cloud import storage as gcs_lib
                client = gcs_lib.Client()
                bucket = client.bucket(GCS_BUCKET)
                gcs_key = output_path.replace(f"gs://{GCS_BUCKET}/", "")
                blob = bucket.blob(gcs_key)
                blob.upload_from_filename(local_output)
                log.info(f"[GCS] Uploaded output: {gcs_key}")
                os.remove(local_output)
            except Exception as ue:
                log.error(f"[GCS] Upload failed: {ue}")
                output_path = local_output

        log.info(f"Job {job_id} completed: {row_count} rows → {output_path}")
        jobs[job_id] = {
            "status": "completed",
            "job_id": job_id,
            "output_path": output_path,
            "row_count": row_count
        }

    except Exception as e:
        import traceback; tb = traceback.format_exc(); log.error(f"Job {job_id} failed: {e}\n{tb}"); open("/tmp/last_error.txt","w").write(tb)
        jobs[job_id] = {"status": "failed", "job_id": job_id, "error": str(e)}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)

# Merge job - birden fazla CSV'yi Spark ile merge et
def merge_csvs_with_spark(spark, input_paths, output_path):
    import time as _time
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor
    _t0 = _time.time()
    GCS_BUCKET = os.getenv("GCS_BUCKET", "schemalabs-prod-us-central1")
    from google.cloud import storage as gcs_lib
    client = gcs_lib.Client()
    bucket = client.bucket(GCS_BUCKET)
    SPARK_SIZE_MB = 50

    def dl(p):
        key = p.replace("./", "").replace("../", "")
        local = f"/tmp/spark_merge_{os.path.basename(key)}"
        bucket.blob(key).download_to_filename(local)
        return local

    local_files = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        futures = {ex.submit(dl, p): p for p in input_paths}
        for f in futures:
            try:
                local_files.append(f.result())
            except Exception as e:
                log.error(f"Failed to download {futures[f]}: {e}")

    if not local_files:
        return 0

    total_mb = sum(os.path.getsize(f) for f in local_files) / (1024*1024)
    log.info(f"[MERGE] {len(local_files)} files downloaded in {_time.time()-_t0:.1f}s, {total_mb:.1f}MB")

    gcs_key = output_path.replace("./", "").replace("../", "")
    local_out = f"/tmp/spark_merged_{os.path.basename(output_path)}.csv"

    if total_mb > SPARK_SIZE_MB:
        log.info(f"[MERGE] Using Spark ({total_mb:.1f}MB > {SPARK_SIZE_MB}MB)")
        from functools import reduce
        dfs = []
        for lf in local_files:
            try:
                df = spark.read.option("header","true").option("inferSchema","true").option("multiLine","true").option("escape",chr(34)).csv(lf)
                df = sanitize_columns(df)
                dfs.append(df)
            except Exception as e:
                log.error(f"[MERGE] Spark load failed {lf}: {e}")
        if not dfs:
            return 0
        merged_spark = reduce(lambda a, b: a.unionByName(b, allowMissingColumns=True), dfs)
        row_count = merged_spark.count()
        tmp_dir = local_out + "_tmp"
        merged_spark.coalesce(1).write.mode("overwrite").option("header","true").csv(tmp_dir)
        import glob, shutil
        parts = glob.glob(tmp_dir + "/part-*.csv")
        if parts:
            shutil.copy(parts[0], local_out)
            shutil.rmtree(tmp_dir)
    else:
        log.info(f"[MERGE] Using pandas ({total_mb:.1f}MB <= {SPARK_SIZE_MB}MB)")
        dfs = [pd.read_csv(f, low_memory=False) for f in local_files]
        merged = pd.concat(dfs, ignore_index=True)
        row_count = len(merged)
        merged.to_csv(local_out, index=False)

    for f in local_files:
        try: os.remove(f)
        except: pass

    bucket.blob(gcs_key).upload_from_filename(local_out)
    os.remove(local_out)
    log.info(f"[GCS] Merged file written: {gcs_key}")
    log.info(f"Merged {len(local_files)} files: {row_count} rows")
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
        pass
    return df
