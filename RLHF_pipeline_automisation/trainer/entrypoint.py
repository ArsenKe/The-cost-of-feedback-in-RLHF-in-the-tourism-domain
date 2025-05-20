from flask import Flask, request, jsonify
import train_dpo 

app = Flask(__name__)

@app.route("/run", methods=["POST"])
def run_job():
    train_dpo.main()        
    return jsonify(status="ok")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
