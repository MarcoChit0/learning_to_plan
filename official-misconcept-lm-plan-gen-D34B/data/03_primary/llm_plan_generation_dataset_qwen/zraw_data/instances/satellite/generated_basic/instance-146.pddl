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
	image0 - mode
	star0 - direction
	planet1 - direction
	star2 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star0)
	(supports instrument1 image0)
	(calibration_target instrument1 star0)
	(supports instrument2 image0)
	(calibration_target instrument2 star0)
	(supports instrument3 image0)
	(calibration_target instrument3 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument4 image0)
	(calibration_target instrument4 star0)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet1)
)
(:goal (and
	(pointing satellite0 star0)
	(pointing satellite1 star2)
	(have_image planet1 image0)
	(have_image star2 image0)
))

)
