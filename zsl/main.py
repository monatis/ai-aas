from sentence_transformers import SentenceTransformer
import redis
import json
import time
import numpy as np
import logging

# Softmax activation function to smoothe cosine similarities for zero shot classification
def softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


def run_zsl():
    logging.warning("Pulling started...")
    while True:
        with queue.pipeline() as pipe:
            pipe.lrange("zsl", 0, 7)
            pipe.ltrim("zsl", 8, -1)
            requests, _ = pipe.execute()


        for r in requests:
            logging.warning(f"Processing: {r}")
            r = json.loads(r)
            embeddings = model.encode(r["texts"] + r["labels"], show_progress_bar=False)
            logging.warning(f"embeddings: {len(embeddings)}")
            label_embeddings = embeddings[len(r["texts"]):]
            results = []
            for i in range(len(r["texts"])):
                dists = []
                for j in range(len(label_embeddings)):
                    dists.append(np.inner(embeddings[i], label_embeddings[j]))

                dists = softmax(np.array(dists))
                idx = np.argmax(dists)
                results.append({'text': r["texts"][i], 'label': r["labels"][idx], 'score': float(dists[idx])})

            queue.set(r["id"], json.dumps(results))

        time.sleep(0.2)



if __name__ == "__main__":
    model = SentenceTransformer("/bert-base-turkish-cased-nli-mean-tokens")
    queue = redis.StrictRedis(host="redis")
    logging.warn("Connected to Redis")

    run_zsl()