class PointConvert {
    constructor(width, height) {
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
    }

    convert(points) {
        return tf.matMul(points, this.convert_marix).arraySync();
    }
}

class Points {
    constructor(points, width, color, pc) {
        points = tf.concat2d([
            points.transpose(),
            tf.ones([1, points.shape[0]])
        ]).transpose();
        this.points = pc.convert(points);

        this.width = width;
        this.color = color;
    }

    plot(graphics) {
        graphics.lineStyle(this.width, this.color);
        graphics.moveTo(this.points[0][0], this.points[0][1]);
        for (let i = 1; i < this.points.length; i++) {
            graphics.lineTo(this.points[i][0], this.points[i][1]);
        }
    }

    scatter(graphics) {
        graphics.lineStyle(this.width, this.color);
        graphics.beginFill(this.color);
        for (let i = 1; i < this.points.length; i++) {
            graphics.drawCircle(this.points[i][0], this.points[i][1], this.width);
        }
        graphics.endFill();
    }
};

class FillRange {
    constructor(x, y, color, pc) {
        var points = tf.stack([
            x,
            y,
            tf.ones([x.shape[0]]),
        ]).transpose();
        this.points = pc.convert(points);

        this.color = color;
    }
    plot(graphics) {
        graphics.beginFill(this.color);
        graphics.lineStyle(1, this.color);
        graphics.moveTo(this.points[0][0], this.points[0][1]);
        for (let i = 1; i < this.points.length; i++) {
            graphics.lineTo(this.points[i][0], this.points[i][1]);
        }
        graphics.closePath();
        graphics.endFill();
    }
}