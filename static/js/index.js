window.HELP_IMPROVE_VIDEOJS = false;

// Interpolation
var INTERP_BASE = "./static/interpolation/stacked_new";
var NUM_INTERP_FRAMES = 104;

var interp_images = [];
function preloadInterpolationImages() {
  for (var i = 0; i <= NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + '/interpolate (' + String(i) + ').jpg';
    interp_images[i] = new Image();
    interp_images[i].src = path;
  }
}

function setInterpolationImage(i) {
  var image = interp_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper').empty().append(image);
}

// Example 1
var EXAMPLE1_BASE = "./static/interpolation/example1";
var NUM_EXAMPLE1_FRAMES = 20;

var example1_images = [];
function preloadExample1Images() {
  for (var i = 0; i <= NUM_EXAMPLE1_FRAMES; i++) {
    var path = EXAMPLE1_BASE + '/visualization (' + String(i) + ').jpg';
    example1_images[i] = new Image();
    example1_images[i].src = path;
  }
}

function setExample1Image(i) {
  var image = example1_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#example1-image-wrapper').empty().append(image);
}

// Example 2
var EXAMPLE2_BASE = "./static/interpolation/example2";
var NUM_EXAMPLE2_FRAMES = 12;

var example2_images = [];
function preloadExample2Images() {
  for (var i = 0; i <= NUM_EXAMPLE2_FRAMES; i++) {
    var path = EXAMPLE2_BASE + '/visualization (' + String(i) + ').jpg';
    example2_images[i] = new Image();
    example2_images[i].src = path;
  }
}

function setExample2Image(i) {
  var image = example2_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#example2-image-wrapper').empty().append(image);
}

// Example 3
var EXAMPLE3_BASE = "./static/interpolation/example3";
var NUM_EXAMPLE3_FRAMES = 19;

var example3_images = [];
function preloadExample3Images() {
  for (var i = 0; i <= NUM_EXAMPLE3_FRAMES; i++) {
    var path = EXAMPLE3_BASE + '/visualization (' + String(i) + ').jpg';
    example3_images[i] = new Image();
    example3_images[i].src = path;
  }
}

function setExample3Image(i) {
  var image = example3_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#example3-image-wrapper').empty().append(image);
}

// Example 4
var EXAMPLE4_BASE = "./static/interpolation/example4";
var NUM_EXAMPLE4_FRAMES = 24;

var example4_images = [];
function preloadExample4Images() {
  for (var i = 0; i <= NUM_EXAMPLE4_FRAMES; i++) {
    var path = EXAMPLE4_BASE + '/visualization (' + String(i) + ').jpg';
    example4_images[i] = new Image();
    example4_images[i].src = path;
  }
}

function setExample4Image(i) {
  var image = example4_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#example4-image-wrapper').empty().append(image);
}

// Example 5
var EXAMPLE5_BASE = "./static/interpolation/example5";
var NUM_EXAMPLE5_FRAMES = 24;

var example5_images = [];
function preloadExample5Images() {
  for (var i = 0; i <= NUM_EXAMPLE5_FRAMES; i++) {
    var path = EXAMPLE5_BASE + '/visualization (' + String(i) + ').jpg';
    example5_images[i] = new Image();
    example5_images[i].src = path;
  }
}

function setExample5Image(i) {
  var image = example5_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#example5-image-wrapper').empty().append(image);
}

// Example 6
var EXAMPLE6_BASE = "./static/interpolation/example6";
var NUM_EXAMPLE6_FRAMES = 21;

var example6_images = [];
function preloadExample6Images() {
  for (var i = 0; i <= NUM_EXAMPLE6_FRAMES; i++) {
    var path = EXAMPLE6_BASE + '/visualization (' + String(i) + ').jpg';
    example6_images[i] = new Image();
    example6_images[i].src = path;
  }
}

function setExample6Image(i) {
  var image = example6_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#example6-image-wrapper').empty().append(image);
}

$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
			slidesToScroll: 6,
			slidesToShow: 6,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    preloadInterpolationImages();

    $('#interpolation-slider').on('input', function(event) {
      setInterpolationImage(this.value);
    });
    setInterpolationImage(0);
    $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES);


    preloadExample1Images();
    $('#example1-slider').on('input', function(event) {
      setExample1Image(this.value);
    });
    setExample1Image(0);
    $('#example1-slider').prop('max', NUM_EXAMPLE1_FRAMES);

    preloadExample2Images();
    $('#example2-slider').on('input', function(event) {
      setExample2Image(this.value);
    });
    setExample2Image(0);
    $('#example2-slider').prop('max', NUM_EXAMPLE2_FRAMES);

    preloadExample3Images();
    $('#example3-slider').on('input', function(event) {
      setExample3Image(this.value);
    });
    setExample3Image(0);
    $('#example3-slider').prop('max', NUM_EXAMPLE3_FRAMES);

    preloadExample4Images();
    $('#example4-slider').on('input', function(event) {
      setExample4Image(this.value);
    });
    setExample4Image(0);
    $('#example4-slider').prop('max', NUM_EXAMPLE4_FRAMES);

    preloadExample5Images();
    $('#example5-slider').on('input', function(event) {
      setExample5Image(this.value);
    });
    setExample5Image(0);
    $('#example5-slider').prop('max', NUM_EXAMPLE5_FRAMES);

    preloadExample6Images();
    $('#example6-slider').on('input', function(event) {
      setExample6Image(this.value);
    });
    setExample6Image(0);
    $('#example6-slider').prop('max', NUM_EXAMPLE6_FRAMES);

    bulmaSlider.attach();

})
