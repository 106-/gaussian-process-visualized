function multivariateGaussian(mean, cov, length) {
    Z = tf.randomNormal([mean.shape[0], length]);
    L = _choleskyDecomposition(cov.arraySync());
    return tf.matMul(L, Z).add(mean);
}

function _choleskyDecomposition(matrix) {
    // Argument "matrix" can be either math.matrix or standard 2D array
    const A = math.matrix(matrix);
    // Matrix A must be symmetric
    console.assert(math.deepEqual(A, math.transpose(A)));

    const n = A.size()[0];
    // Prepare 2D array with 0
    const L = new Array(n).fill(0).map(_ => new Array(n).fill(0));

    d3.range(n).forEach(i => {
        d3.range(i + 1).forEach(k => {
            const s = d3.sum(d3.range(k).map(j => L[i][j] * L[k][j]));
            L[i][k] = i === k ? math.sqrt(A.get([k, k]) - s) : 1 / L[k][k] * (A.get([i, k]) - s);
        })
    });
    return L;
}

class GaussianProcess {
    constructor(x_train, y_train, kernel) {
        var distance = this._distanceMatrix(x_train, x_train);
        this.kernel = kernel;
        this.x_train = x_train;
        this.y_train = y_train;
        this.K = this.kernel.forward(distance, true);
        this.K_inv = tf.tensor(math.inv(this.K.arraySync()));
    }

    sampling(length) {
        return multivariateGaussian(tf.zeros([this.K.shape[0], 1]), this.K, length);
    }

    predict(x_pred) {
        var K_star = this.kernel.forward(this._distanceMatrix(this.x_train, x_pred), false);
        var K_starstar = this.kernel.forward(this._distanceMatrix(x_pred, x_pred), false);

        var K_star_K_inv = tf.matMul(K_star.transpose(), this.K_inv);
        var y_pred_mu = tf.matMul(K_star_K_inv, this.y_train).squeeze().arraySync();
        var K_pred = K_starstar.sub(tf.matMul(K_star_K_inv, K_star)).arraySync();

        var y_pred_sigma = new Array(x_pred.shape[0]).fill(0);
        d3.range(x_pred.shape[0]).forEach(i => { y_pred_sigma[i] = K_pred[i][i]; });

        return [y_pred_mu, y_pred_sigma];
    }

    // https://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow
    _distanceMatrix(x1, x2) {
        const a = x1.expandDims(1);
        const b = x2.expandDims(0);
        const c = tf.squaredDifference(a, b).sum(2);
        return c;
    }
}

// 学習できないガウス過程モデル
class PrimitiveGaussianProcess {
    constructor(data, kernel) {
        var distance = this._distanceMatrix(data);
        this.kernel = kernel;
        this.K = this.kernel.forward(distance);
        this.K_inv = tf.tensor(math.inv(this.K.arraySync()));
    }

    sampling(length) {
        return multivariateGaussian(tf.zeros([this.K.shape[0], 1]), this.K, length);
    }

    // https://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow
    _distanceMatrix(data) {
        const a = data.expandDims(1);
        const b = data.expandDims(0);
        return tf.squaredDifference(a, b).sum(2);
    }
}

class RBFKernel {
    constructor(theta1, theta2, theta3) {
        this.theta1 = theta1;
        this.theta2 = theta2;
        this.theta3 = theta3;
    }

    forward(distance, add_likelihood) {
        var K = tf.exp(distance.mul(-1).div(this.theta2)).mul(this.theta1);

        if (add_likelihood) {
            var likelihood = tf.eye(distance.shape[0]).mul(this.theta3);
            K = K.add(likelihood);
        }
        return K;
    }
}