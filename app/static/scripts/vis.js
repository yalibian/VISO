// Populate a grid of n×m values where -2 ≤ x ≤ 2 and -2 ≤ y ≤ 1.


let svg = d3.select("#vis");

let rect = svg.node()
        .getBoundingClientRect(),
    width = rect.width,
    height = rect.height;


let bar_height = d3.select('#visualization-bar').node().getBoundingClientRect().height;
let selectedOpt = [];
let selectedObj = "flower";
let selectedEpoch = [];
let selectedLearningRate = [];
let selectedDecayRate = [];

let pos = [-6, -6];
let selectedX = -6;
let selectedY = -6;
let bounds = [[-6, 6], [-6, 6]];

Array.prototype.remove = function () {

    let what, a = arguments, L = a.length, ax;
    while (L && this.length) {
        what = a[--L];
        while ((ax = this.indexOf(what)) !== -1) {
            this.splice(ax, 1);
        }
    }
    return this;
};

let view2posX = d3.scaleLinear()
    .domain([0, width])
    .range(bounds[0]);

let view2posY = d3.scaleLinear()
    .domain([0, height])
    .range(bounds[1]);

function drawCircle(x, y, size) {
    // console.log('Drawing circle at', x, y, size);
    svg.selectAll("circle")
        .remove();

    svg.append("circle")
        .attr('class', 'click-circle')
        .attr('fill', 'yellow')
        .attr("cx", x)
        .attr("cy", y)
        .attr("r", size);
}


$(document).ready(function () {

    svg.on('click', d => {

        console.log('------------');
        [x, y] = [d3.event.x, d3.event.y];
        // console.log(x, y);
        console.log(bar_height);
        console.log(d3.event.clientX, d3.event.clientY - bar_height);
        pos[0] = view2posX(x).toFixed(2);
        pos[1] = view2posY(y - bar_height).toFixed(2);
        drawCircle(x, y - bar_height, 5.5);
        // drawCircle(0, 0, 5.5);
        $('#StartPoint').html('(' + pos[0] + ', ' + pos[1] + ')');
    });

    let obj = {
        "opt": [],
        "width": width,
        "height": height,
        "obj": "flower",
        "epoch": 1000,
        "rate": 1000,
        "reg": 0.01,
        "customize": false,
        "pos": pos,
        "X": [-6, 6],
        "Y": [-6, 6]
    };

    d3.request('/training')
        .mimeType("text/csv")
        .post(JSON.stringify(obj), function (error, y, z) {
            let data = JSON.parse(y.response).res;
            let values = data.values;
            delete data.values;
            updateVis(values, data);
        });

    $('#play-pause-button')
        .click(function (hello) {
            var obj = {
                "opt": selectedOpt,
                "width": width,
                "height": height,
                "obj": selectedObj,
                "epoch": selectedEpoch,
                "rate": selectedLearningRate,
                "reg": selectedDecayRate,
                "customize": false,
                "pos": pos,
                "X": [-6, 6],
                "Y": [-6, 6]
            };

            if ($('#myModal').is(":visible")) {
                obj.X = [$('#x1').val(), $('#x2').val()];
                obj.Y = [$('#y1').val(), $('#y2').val()];
                obj.customize = true;
                obj.obj = $('#objective-function-python').val();
            }

            d3.request('/training')
                .mimeType("text/csv")
                .post(JSON.stringify(obj), function (error, y, z) {
                    console.log(y.response);
                    let data = JSON.parse(y.response).res;
                    let values = data.values;
                    delete data.values;
                    updateVis(values, data);
                });
        });

    $('#myModal').hide();

    $('#optimizer').multiselect({

        onChange: function (option, checked, select) {

            let opt = $(option).val();
            if (selectedOpt.includes(opt)) {
                selectedOpt.remove(opt);
            } else {
                selectedOpt.push(opt);
            }
        }
    });


    $('#epoch').multiselect({

            onChange: function (option, checked, select) {

                let epoch = $(option).val();
                if (selectedEpoch.includes(epoch)) {
                    selectedEpoch.remove(epoch);
                } else {
                    selectedEpoch.push(epoch);
                }
            }
        }
    );

    $('#objective').multiselect({

        onChange: function (option, checked, select) {

            $(".ui-objective .multiselect-selected-text").html($(option).html());


            let obj = $(option).val();

            if (obj === "customize") {
                $('#myModal').show();
            } else {
                selectedObj = obj;
                $('#myModal').hide();
            }
        }
    });

    $('#learningRate').multiselect({
        onChange: function (option, checked, select) {

            let rate = $(option).val();
            if (selectedLearningRate.includes(rate)) {
                selectedLearningRate.remove(rate);
            } else {
                selectedLearningRate.push(rate);
            }

        }
    });

    $('#regularizations').multiselect({

        onChange: function (option, checked, select) {
            // console.log($(option).val());
        }
    });
    $('#dacayRate').multiselect({

            onChange: function (option, checked, select) {
                let opt = $(option).val();
                if (selectedDecayRate.includes(opt)) {
                    selectedDecayRate.remove(opt);
                } else {
                    selectedDecayRate.push(opt);
                }

            }
        }
    );
});

// function updateObjectiveFunction(x) {
//     ObjectveFunction
//     objectiveFun = x;
//     updateVis(str2fun(objectiveFun));
// }

// function str2fun(objectiveFun) {
//     switch (objectiveFun) {
//         case "goldsteinPrice":
//             return goldsteinPrice;
//         case "flower":
//             return flower;
//         case "himmelblau":
//             return himmelblau;
//         case "banana":
//             return banana;
//         case "matyas":
//             return matyas;
//     }
//
// }


// 是不是map的问题啊？？？
function updateVis(values, paths) {

    svg.selectAll("*").remove();
    let thresholds = [0.0025, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.2, 0.5, 0.8, 2.0, 5.00, 10, 25.00, 50, 100, 150.0, 200.0, 250.0, 300.0, 400.0, 500];


    // draw contours
    let contours = d3.contours()
        .size([width, height])
        .thresholds(thresholds);

    let color = d3.scaleLog()
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


    // draw lines
    console.log('draw lines');
    console.log(paths);

    var x = d3.scaleLinear()
        .rangeRound([0, width]);

    x.domain([-6, 6]);

    var y = d3.scaleLinear()
        .rangeRound([0, height]);
    y.domain([-6, 6]);

    console.log('------------------ location: -------------------')
    console.log(x(-6)); // 0
    console.log(x(0))
    console.log(x(6)); // 1440


    let line = d3.line()
    // .interpolate("cardinal")
        .x(function (d) {
            return x(d[0]);
        })
        .y(function (d) {
            return y(d[1]);
        });


    // console.log();
    // let c10 = d3.scaleCategory10;
    let c10 = d3.scaleOrdinal(d3.schemeCategory10);


    Object.keys(paths).forEach(function (key, i) {

        console.log(paths[key]);


        // svg.append("path")
        //     .datum(paths[key])
        //     .attr("fill", "none")
        //     .attr("stroke", c10(i))
        //     .attr("stroke-linejoin", "round")
        //     .attr("stroke-linecap", "round")
        //     .attr("stroke-width", 4.5)
        //     .attr("d", line);

        let transition = function (path) {
            path.transition()
                .duration(2000)
                .attrTween("stroke-dasharray", tweenDash);
        };

        let tweenDash = function () {
            let l = this.getTotalLength(),
                i = d3.interpolateString("0," + l, l + "," + l);
            return function (t) {
                return i(t);
            };
        };

        let path = svg.append("path")
            .attr("d", line(paths[key]))
            .attr("stroke", c10(i))
            .attr("stroke-width", "3")
            .attr("fill", "none")
            .call(transition);

        // let totalLength = path.node().getTotalLength();

        // path
        //     .attr("stroke-dasharray", totalLength + " " + totalLength)
        //     .attr("stroke-dashoffset", totalLength)
        //     .transition()
        //     .duration(2000)
        //     .ease("linear")
        //     .attr("stroke-dashoffset", 0);

    });


    // let x = d3.scaleTime().range([0, width]),
    //     y = d3.scaleLinear().range([height, 0]),
    //     z = d3.scaleOrdinal(d3.schemeCategory10);
    //
    // let line = d3.line()
    //     .curve(d3.curveBasis)
    //     .x(function (d) {
    //         return x(d.date);
    //     })
    //     .y(function (d) {
    //         return y(d.temperature);
    //     });
    //
    // let optpaths = svg.selectAll(".optpath")
    //     .data(paths)
    //     .enter()
    //     .append("g")
    //     .attr("class", "optpath");
    //
    // optpaths.append("path")
    //     .attr("class", "line")
    //     .attr("d", function (d) {
    //         return line(d.values);
    //     })
    //     .style("stroke", function (d) {
    //         return z(d.id);
    //     });

    // city.append("text")
    //     .datum(function (d) {
    //         return {id: d.id, value: d.values[d.values.length - 1]};
    //     })
    //     .attr("transform", function (d) {
    //         return "translate(" + x(d.value.date) + "," + y(d.value.temperature) + ")";
    //     })
    //     .attr("x", 3)
    //     .attr("dy", "0.35em")
    //     .style("font", "10px sans-serif")
    //     .text(function (d) {
    //         return d.id;
    //     });


}


// calculate the values based on the function, weight, and height
function fun2mat(f) {

    let scaleX = d3.scaleLinear()
        .domain([0, width])
        .range([0.0, 1.0]);

    let scaleY = d3.scaleLinear()
        .domain([0, height])
        .range([0.0, 1.0]);

    let values = new Array(width * height);
    // let maxV = 0;

    for (var j = 0.0, k = 0; j < height; ++j) {
        for (var i = 0.0; i < width; ++i, ++k) {
            values[k] = f(scaleX(i), scaleY(j));
            // if (values[k] > maxV) {
            //     maxV = values[k];
            // }
        }
    }

    let fun = {"values": values};

    return fun;
}


function flower(x, y) {

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


