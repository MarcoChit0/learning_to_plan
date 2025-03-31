(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	infrared0 - mode
	image2 - mode
	image1 - mode
	star1 - direction
	groundstation0 - direction
	star2 - direction
	star3 - direction
	phenomenon4 - direction
)
(:init
	(supports instrument0 image2)
	(calibration_target instrument0 star1)
	(supports instrument1 image1)
	(supports instrument1 image2)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument2 image2)
	(calibration_target instrument2 star1)
	(supports instrument3 infrared0)
	(supports instrument3 image2)
	(supports instrument3 image1)
	(calibration_target instrument3 star1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon4)
	(supports instrument4 infrared0)
	(supports instrument4 image2)
	(supports instrument4 image1)
	(calibration_target instrument4 groundstation0)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
)
(:goal (and
	(pointing satellite0 star2)
	(have_image star2 image1)
	(have_image star3 image2)
	(have_image phenomenon4 image1)
))

)
