var SCREEN_WIDTH = window.innerWidth;
var SCREEN_HEIGHT = window.innerHeight;

class RandomFunction {
    constructor() {
        this.param = tf.randomUniform([4], 1, 3).arraySync();
        this.pi = 3.1415926535;
    }
    target(x, scale) {
        var y1 = tf.sin(x.mul(this.pi / this.param[0]));
        var y2 = tf.cos(x.mul(this.pi / this.param[1]));
        var y3 = tf.sin(x.mul(this.pi / this.param[2]));
        var y4 = tf.cos(x.mul(this.pi / this.param[3]));
        var y5 = tf.randomNormal([x.shape[0]]).mul(scale);
        return y1.add(y2).add(y3).add(y4).add(y5).div(2);
    }
}

init();

function init() {
    // Pixiアプリケーション生成
    let app = new PIXI.Application({
        width: SCREEN_WIDTH,
        height: SCREEN_HEIGHT,
        backgroundColor: 0xD6D6D6,
        autoDensity: true,
        antialias: true,
    });
    const graphics = new PIXI.Graphics();
    let el = document.getElementById("canvas");
    el.appendChild(app.view);

    pc = new PointConvert(SCREEN_WIDTH, SCREEN_HEIGHT);

    // 真の関数の作成
    rf = new RandomFunction();
    twopi = 2 * 3.1415
    x_ground_truth = tf.linspace(-twopi, twopi, 1000);
    y_ground_truth = rf.target(x_ground_truth, 0);
    array = tf.stack([x_ground_truth, y_ground_truth]).transpose();
    gt_line = new Points(array, 3, 0x4F4F4F, pc);

    // 観測値の作成
    x_train = tf.randomNormal([50], 0, 2.5);
    y_train = rf.target(x_train, 0.25);
    array = tf.stack([x_train, y_train]).transpose();
    observed = new Points(array, 3, 0x505050, pc);

    // ガウス過程モデルの作成
    // gp = new GaussianProcess(x_train.expandDims(1), y_train.expandDims(1), new RBFKernel(1., 0.3, 0.5));
    // gp = new GaussianProcess(x_train.expandDims(1), y_train.expandDims(1), new ExponentialKernel(2.5, 0.5));
    gp = new GaussianProcess(x_train.expandDims(1), y_train.expandDims(1), new MaternKernel(0.75, 1.0));
    x_pred = tf.linspace(-twopi, twopi, 200);
    [mu, sigma] = gp.predict(x_pred.expandDims(1));

    array = tf.stack([x_pred, mu]).transpose();
    predicted_mean = new Points(array, 2, 0xE8E8E8, pc);

    var sigma_range = tf.tensor(sigma).mul(3.0)
    var x = tf.concat([x_pred, tf.reverse(x_pred)]);
    var y = tf.concat([tf.add(mu, sigma_range), tf.reverse(tf.sub(mu, sigma_range))]);
    confidence = new FillRange(x, y, 0xADADAD, pc);

    // 描画
    confidence.plot(graphics);
    // predicted_mean.plot(graphics);
    gt_line.plot(graphics);
    observed.scatter(graphics);

    app.stage.addChild(graphics);
}


function animate(delta) {

}


