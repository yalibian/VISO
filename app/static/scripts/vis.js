// Populate a grid of n×m values where -2 ≤ x ≤ 2 and -2 ≤ y ≤ 1.

// var svg = d3.select("#vis"),
//      width = +svg.attr("width"),
//      height = +svg.attr("height");

const svg = d3.select('#vis');
const width = parseInt(svg.style("width"), 10);
const height = parseInt(svg.style("height"), 10);


console.log(width + " " + height);
var n = 240, m = Math.floor(n *  width / height);
var m = 170, n = 240;

console.log(n + " " + m);


var values = new Array(n * m);
for (var j = 0.5, k = 0; j < m; ++j) {
    for (var i = 0.5; i < n; ++i, ++k) {
        values[k] = goldsteinPrice(i / n * 4 - 2, 1 - j / m * 3);
    }
}


var thresholds = d3.range(1, 21)
    .map(function (p) {
        return Math.pow(2, p);
    });

var contours = d3.contours()
    .size([n, m])
    .thresholds(thresholds);

var color = d3.scaleLog()
    .domain(d3.extent(thresholds))
    .interpolate(function () {
        return d3.interpolateYlGnBu;
    });

svg.selectAll("path")
    .data(contours(values))
    .enter().append("path")
    .attr("d", d3.geoPath(d3.geoIdentity().scale(width / n)))
    .attr("fill", function (d) {
        return color(d.value);
    });

// See https://en.wikipedia.org/wiki/Test_functions_for_optimization
function goldsteinPrice(x, y) {
    return (1 + Math.pow(x + y + 1, 2) * (19 - 14 * x + 3 * x * x - 14 * y + 6 * x * x + 3 * y * y))
        * (30 + Math.pow(2 * x - 3 * y, 2) * (18 - 32 * x + 12 * x * x + 48 * y - 36 * x * y + 27 * y * y));
}
