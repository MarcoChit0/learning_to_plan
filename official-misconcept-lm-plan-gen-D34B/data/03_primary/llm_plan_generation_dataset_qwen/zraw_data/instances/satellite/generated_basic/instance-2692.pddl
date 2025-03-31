(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite1 - satellite
	instrument4 - instrument
	image3 - mode
	infrared0 - mode
	image1 - mode
	image2 - mode
	star1 - direction
	star2 - direction
	star0 - direction
	star3 - direction
	planet4 - direction
)
(:init
	(supports instrument0 image2)
	(supports instrument0 infrared0)
	(supports instrument0 image3)
	(calibration_target instrument0 star0)
	(supports instrument1 infrared0)
	(supports instrument1 image1)
	(calibration_target instrument1 star2)
	(supports instrument2 image1)
	(supports instrument2 image3)
	(calibration_target instrument2 star0)
	(supports instrument3 image1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument4 infrared0)
	(supports instrument4 image1)
	(calibration_target instrument4 star3)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
)
(:goal (and
	(pointing satellite1 planet4)
	(have_image planet4 infrared0)
))

)
