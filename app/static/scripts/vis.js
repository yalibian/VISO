// Populate a grid of n×m values where -2 ≤ x ≤ 2 and -2 ≤ y ≤ 1.

var svg = d3.select("#vis");
     // width = +svg.attr("width"),
     // height = +svg.attr("height");


var rect = svg.node().getBoundingClientRect(),
    width = rect.width,
    height = rect.height;


var objectiveFun = "flower";
// const svg = d3.select('#vis');

// const width = parseInt(svg.style("width"), 10);
// const height = parseInt(svg.style("height"), 10);

console.log(width, height);

updateVis(str2fun(objectiveFun));


$(document).ready(function () {
    $('#optimizer').multiselect();
    $('#epoch').multiselect();
    $('#objective').multiselect({

        onChange: function (option, checked, select) {
            // console.log($(option).innerHTML);
            // console.log($(option).innerHTML());
            // console.log(option.innerHTML());
            // $(".multiselect-selected-text").html($(option).html());
            // $(".ui-objective").
            // $(".ui-objective .multiselect-selected-text").text($(option).html());
            $(".ui-objective .multiselect-selected-text").html($(option).html());

            updateObjectiveFunction($(option).val());

            // stopHere(x);
        }
    });
    $('#learningRate').multiselect();
    $('#regularizations').multiselect();
    $('#regularRate').multiselect();
});

function updateObjectiveFunction(x) {
    // ObjectveFunction
    objectiveFun = x;
    updateVis(str2fun(objectiveFun));
}

function str2fun(objectiveFun) {
    switch (objectiveFun) {
        case "goldsteinPrice":
            return goldsteinPrice;
        case "flower":
            return flower;
        case "himmelblau":
            return himmelblau;
        case "banana":
            return banana;
        case "matyas":
            return matyas;
    }

}

function updateVis(f) {

    svg.selectAll("*").remove();

    let fun = fun2mat(f);
    let maxV = fun.max;
    let values = fun.values;


    var thresholds = d3.range(21, 1, -1)
        .map(function (p) {
            return Math.sqrt(p, 2);
        });


    console.log(thresholds);

    var contours = d3.contours()
        .size([width, height])
        .thresholds(thresholds);

    var color = d3.scaleLog()
        .domain(d3.extent(thresholds))
        .interpolate(function () {
            return d3.interpolateYlGnBu;
        });


    svg.selectAll("path")
        .data(contours(values))
        .enter()
        .append("path")
        .attr("d", d3.geoPath())
        .attr("fill", function (d) {
            return color(d.value);
        });

}


// calculate the values based on the function, weight, and height
function fun2mat(f) {


    // var n = 240, m = Math.floor(n * width / height);
    // var m = 170, n = 240;
    // var m = width, n = height;

    let scaleX = d3.scaleLinear()
        .domain([0, width])
        .range([0.0, 1.0]);

    let scaleY = d3.scaleLinear()
        .domain([0, height])
        .range([0.0, 1.0]);

    var values = new Array(width * height);
    var maxV = 0

    for (var j = 0.0, k = 0; j < height; ++j) {
        for (var i = 0.0; i < width; ++i, ++k) {
            values[k] = f(scaleX(i), scaleY(j));
            if (values[k] > maxV){
                maxV = values[k];
            }
        }
    }

    let fun = {"values": values, max: maxV };

    return fun;
}


// See https://en.wikipedia.org/wiki/Test_functions_for_optimization
// function goldsteinPrice(x, y) {
//     return (1 + Math.pow(x + y + 1, 2) * (19 - 14 * x + 3 * x * x - 14 * y + 6 * x * x + 3 * y * y))
//         * (30 + Math.pow(2 * x - 3 * y, 2) * (18 - 32 * x + 12 * x * x + 48 * y - 36 * x * y + 27 * y * y));
// }

function flower(x, y) {

    // the boud
    // let X = [-6, 6], Y = [-6, 6];
    let scaleX = d3.scaleLinear()
        .domain([0.0, 1.0])
        .range([-6.0, 6.0]);

    let scaleY = d3.scaleLinear()
        .domain([0.0, 1.0])
        .range([-6.0, 6.0]);

    x = scaleX(x);
    y = scaleY(y);
    return (x * x) + (y * y) + x * Math.sin(y) + y * Math.sin(x);
}

function himmelblau(x, y) {

    let scaleX = d3.scaleLinear()
        .domain([0.0, 1.0])
        .range([-6.0, 6.0]);

    let scaleY = d3.scaleLinear()
        .domain([0.0, 1.0])
        .range([-6.0, 6.0]);

    x = scaleX(x);
    y = scaleY(y);

    return Math.pow(x * x - 11, 2) + Math.pow(x + y * y - 7, 2);
}


function banana(x, y) {

    let scaleX = d3.scaleLinear()
        .domain([0.0, 1.0])
        .range([-6.0, 6.0]);

    let scaleY = d3.scaleLinear()
        .domain([0.0, 1.0])
        .range([-2.0, 2.0]);

    x = scaleX(x);
    y = scaleY(y);

    return Math.pow(1 - x, 2) + 100 * Math.pow(y - x * x, 2);
}


function matyas(x, y) {

     let scaleX = d3.scaleLinear()
        .domain([0.0, 1.0])
        .range([-10.0, 10.0]);

    let scaleY = d3.scaleLinear()
        .domain([0.0, 1.0])
        .range([-10.0, 10.0]);

    x = scaleX(x);
    y = scaleY(y);

    return 0.26 * (x * x + y * y) + 0.48 * x * y;
}


