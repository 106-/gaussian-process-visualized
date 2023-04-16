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
    if (interactive) {
        app.stage.interactive = true;
        app.stage.hitArea = app.screen;
    }

    let el = document.getElementById("canvas");
    el.appendChild(app.view);

    pc = new PointConvert(SCREEN_WIDTH, SCREEN_HEIGHT);

    // 真の関数の作成
    rf = new RandomFunction();
    twopi = 2 * 3.1415
    x_ground_truth = tf.linspace(-twopi, twopi, 1000);
    y_ground_truth = rf.target(x_ground_truth, 0);
    ground_truth_array = tf.stack([x_ground_truth, y_ground_truth]).transpose();

    // 観測値の作成
    x_train = tf.randomNormal([initial_x_length], 0, 2.5);
    y_train = rf.target(x_train, 0.25);
    observed_array = tf.stack([x_train, y_train]).transpose();

    // ガウス過程モデルの作成
    x_pred = tf.linspace(-twopi, twopi, 200);
    var mu, sigma;
    if (x_train.shape[0] != 0) {
        gp = new GaussianProcess(x_train.expandDims(1), y_train.expandDims(1), new MaternKernel(0.75, 1.0));
        [mu, sigma] = gp.predict(x_pred.expandDims(1));
    } else {
        mu = tf.zeros([x_pred.shape[0]]);
        sigma = tf.ones([x_pred.shape[0]]).arraySync();
    }


    confidence = new FillRange(x_pred, mu, sigma, 3.0, 0xADADAD, pc);
    ground_truth_line = new Points(ground_truth_array, 3, 0x4F4F4F, pc);
    observed = new Points(observed_array, 9, 0x505050, pc);

    confidence_g = new PIXI.Graphics();
    ground_truth_g = new PIXI.Graphics();
    observed_g = new PIXI.Graphics();

    // 描画
    confidence.plot(confidence_g);
    ground_truth_line.plot(ground_truth_g);
    if (x_train.shape[0] > 0) {
        observed.scatter(observed_g);
    }

    app.stage.addChild(confidence_g);
    app.stage.addChild(ground_truth_g);
    app.stage.addChild(observed_g);

    if (show_mean) {
        predicted_mean_array = tf.stack([x_pred, mu]).transpose();
        predicted_mean = new Points(predicted_mean_array, 2, 0xE8E8E8, pc);
        predicted_mean_g = new PIXI.Graphics();
        predicted_mean.plot(predicted_mean_g);
        app.stage.addChild(predicted_mean_g);
    }

    window.app = app;
    app.stage.on('pointerdown', (e) => {
        var plot_x = e.data.global.x;
        var plot_y = e.data.global.y;
        var point = pc.reverse_convert([[plot_x, plot_y, 1]]).arraySync();
        var x = point[0][0];
        var y = point[0][1];

        x_train = tf.concat1d([x_train, [x]]);
        y_train = tf.concat1d([y_train, [y]]);
        x_pred = tf.linspace(-twopi, twopi, 200);

        gp = new GaussianProcess(x_train.expandDims(1), y_train.expandDims(1), new MaternKernel(0.75, 1.0));
        [mu, sigma] = gp.predict(x_pred.expandDims(1));

        array = tf.stack([x_train, y_train]).transpose();
        observed = new Points(array, 9, 0x505050, pc);

        array = tf.stack([x_pred, mu]).transpose();
        predicted_mean = new Points(array, 2, 0xE8E8E8, pc);
        confidence = new FillRange(x_pred, mu, sigma, 3.0, 0xADADAD, pc);


        confidence_g.clear();
        observed_g.clear();
        confidence.plot(confidence_g);
        observed.scatter(observed_g);

        if (show_mean) {
            predicted_mean_g.clear();
            predicted_mean.plot(predicted_mean_g);
        }
    });

}
