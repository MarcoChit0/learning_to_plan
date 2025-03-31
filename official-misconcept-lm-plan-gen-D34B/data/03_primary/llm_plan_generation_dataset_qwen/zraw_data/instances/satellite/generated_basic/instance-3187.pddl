(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	image0 - mode
	image1 - mode
	star0 - direction
	planet1 - direction
	planet2 - direction
)
(:init
	(supports instrument0 image1)
	(calibration_target instrument0 star0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet1)
	(supports instrument1 image0)
	(supports instrument1 image1)
	(calibration_target instrument1 star0)
	(supports instrument2 image0)
	(supports instrument2 image1)
	(calibration_target instrument2 star0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet2)
	(supports instrument3 image0)
	(supports instrument3 image1)
	(calibration_target instrument3 star0)
	(on_board instrument3 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet1)
)
(:goal (and
	(pointing satellite0 planet2)
	(have_image planet1 image0)
	(have_image planet2 image0)
))

)
