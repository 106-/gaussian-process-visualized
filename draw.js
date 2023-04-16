class PointConvert {
    constructor(width, height) {
        this.resize(width, height);
    }

    convert(points) {
        return tf.matMul(points, this.convert_marix);
    }

    reverse_convert(points) {
        return tf.matMul(points, this.rev_convert_matrix);
    }

    resize(width, height) {
        var scale_width = 6;
        var x_mag = 1 / (scale_width * 2);
        var y_mag = 1 / (scale_width * 2 * (height / width));

        var affine_matrix = [
            // 上下反転
            new tf.tensor([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, 1]
            ]),
            // 縮小
            new tf.tensor([
                [x_mag, 0, 0],
                [0, y_mag, 0],
                [0, 0, 1]
            ]),
            // 原点を移動
            new tf.tensor([
                [1, 0, 0],
                [0, 1, 0],
                [0.5, 0.5, 1]
            ]),
            // 画面サイズに変形
            new tf.tensor([
                [width, 0, 0],
                [0, height, 0],
                [0, 0, 1]
            ]),
        ];

        // 行列をまとめる
        this.convert_marix = tf.eye(3);
        for (var i = 0; i < affine_matrix.length; i++) {
            this.convert_marix = tf.matMul(this.convert_marix, affine_matrix[i]);
        };
        this.rev_convert_matrix = tf.tensor(math.inv(this.convert_marix.arraySync()));
    }
}

class Points {
    constructor(points, width, color, pc) {
        this.points = tf.concat2d([
            points,
            tf.ones([points.shape[0], 1])
        ], 1);

        this.width = width;
        this.color = color;
        this.pc = pc;
    }

    plot(graphics) {
        var pp = this.pc.convert(this.points).arraySync();
        graphics.lineStyle(this.width, this.color);
        graphics.moveTo(pp[0][0], pp[0][1]);
        for (let i = 0; i < pp.length; i++) {
            graphics.lineTo(pp[i][0], pp[i][1]);
        }
    }

    scatter(graphics) {
        var pp = this.pc.convert(this.points).arraySync();
        graphics.beginFill(this.color);
        for (let i = 0; i < pp.length; i++) {
            graphics.drawCircle(pp[i][0], pp[i][1], this.width);
        }
        graphics.endFill();
    }
};

class FillRange {
    constructor(x, mu, sigma, range, color, pc) {
        var sigma_range = tf.tensor(sigma).mul(range);
        this.points = tf.stack([
            tf.concat([x, tf.reverse(x)]),
            tf.concat([tf.add(mu, sigma_range), tf.reverse(tf.sub(mu, sigma_range))]),
            tf.ones([x.shape[0] * 2]),
        ]).transpose();
        this.pc = pc;
        this.color = color;
    }

    plot(graphics) {
        var pp = this.pc.convert(this.points).arraySync();
        graphics.beginFill(this.color);
        graphics.lineStyle(1, this.color);
        graphics.moveTo(pp[0][0], pp[0][1]);
        for (let i = 0; i < pp.length; i++) {
            graphics.lineTo(pp[i][0], pp[i][1]);
        }
        graphics.closePath();
        graphics.endFill();
    }
}