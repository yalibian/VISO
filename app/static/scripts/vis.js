// Populate a grid of n×m values where -2 ≤ x ≤ 2 and -2 ≤ y ≤ 1.


let svg = d3.select("#vis");

let rect = svg.node()
        .getBoundingClientRect(),
    width = rect.width,
    height = rect.height;


let objectiveFun = "flower";
let selectedOpt = [];
let selectedObj = [];
let selectedEpoch = [];
let selectedLearningRate = [];

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


updateVis(str2fun(objectiveFun));


$(document).ready(function () {
    $('#play-pause-button')
        .click(function (hello) {


            var obj = {"opt": selectedOpt, "width": width, "height": height, "obj": objectiveFun, "epoch": selectedEpoch, "rate": selectedLearningRate, "customize": false};

            if($('#myModal').is(":visible")){
                console.log();
                obj.X = [$('#x1').val(), $('#x2').val()];
                obj.Y = [ $('#y1').val(),  $('#y2').val()];
                obj.customize = true;
                obj.obj = $('#objective-function-python').val() ;
            }

            d3.request('/training')
                .mimeType("text/csv")
                .post(JSON.stringify(obj), updateVis);
            console.log("Clicked the Play-pause-button");
        });

    $('#myModal').hide();

    $('#optimizer').multiselect({

        onChange: function (option, checked, select) {
            // console.log($(option).val());
            // console.log(select);
            // console.log(checked);

            let opt = $(option).val();
            if (selectedOpt.includes(opt)) {
                selectedOpt.remove(opt);
            } else {
                selectedOpt.push(opt);
            }

            console.log(selectedOpt);


        }
    });

    $('#epoch').multiselect(
        {

            onChange: function (option, checked, select) {
                console.log($(option).val());

                let epoch = $(option).val();

                if (selectedEpoch.includes(epoch)) {
                    selectedEpoch.remove(epoch);
                } else {
                    selectedEpoch.push(epoch);
                }

                console.log(selectedEpoch);


            }
        }
    );

    $('#objective').multiselect({

        onChange: function (option, checked, select) {
            // console.log($(option).innerHTML);
            // console.log($(option).innerHTML());
            // console.log(option.innerHTML());
            // $(".multiselect-selected-text").html($(option).html());
            // $(".ui-objective").
            // $(".ui-objective .multiselect-selected-text").text($(option).html());
            $(".ui-objective .multiselect-selected-text").html($(option).html());


            let obj = $(option).val();

            if (obj === "customize") {

                console.log("In customize");
                // $('#myModal').on('shown.bs.modal', function () {
                //     $('#myInput').trigger('focus')
                // })
                $('#myModal').show();

            } else {
                if (selectedObj.includes(obj)) {
                    selectedObj.remove(obj);
                } else {
                    selectedObj.push(obj);
                }

                $('#myModal').hide();

            }


            updateObjectiveFunction($(option).val());
        }
    });

    $('#learningRate').multiselect({
        onChange: function (option, checked, select) {
            // console.log($(option).val());

            let rate = $(option).val();
            if (selectedLearningRate.includes(rate)) {
                selectedLearningRate.remove(rate);
            } else {
                selectedLearningRate.push(rate);
            }


            console.log(selectedLearningRate);

        }
    });

    $('#regularizations').multiselect(
        {

            onChange: function (option, checked, select) {
                console.log($(option).val());
            }
        }
    );
    $('#regularRate').multiselect(
        {

            onChange: function (option, checked, select) {
                console.log($(option).val());
            }
        }
    );
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


    // var thresholds = d3.range(1, 20)
    //     .map(function (p) {
    // return powerScale(p);
    // return Math.sqrt(p, 2);
    // });

    var thresholds = [0.0025, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.2, 0.5, 0.8, 2.0, 5.00, 10, 25.00, 50, 100, 150.0, 200.0, 250.0, 300.0, 400.0, 500];
    // thresholds=[0.05, 0.1, 0.2, 0.5, 1.0, 1, 1.4142135623730951, 1.7320508075688772, 2, 2.23606797749979, 2.449489742783178, 2.6457513110645907, 2.8284271247461903, 3, 3.1622776601683795, 3.3166247903554, 3.4641016151377544, 3.605551275463989, 3.7416573867739413, 3.872983346207417, 4, 4.123105625617661, 4.242640687119285, 4.358898943540674];

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


