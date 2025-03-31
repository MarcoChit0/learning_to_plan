(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	image0 - mode
	image2 - mode
	image3 - mode
	thermograph1 - mode
	star2 - direction
	groundstation3 - direction
	groundstation4 - direction
	star0 - direction
	star1 - direction
	planet5 - direction
	star6 - direction
	planet7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 image3)
	(supports instrument0 image0)
	(calibration_target instrument0 star1)
	(supports instrument1 image0)
	(calibration_target instrument1 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation4)
	(supports instrument2 image3)
	(supports instrument2 image2)
	(calibration_target instrument2 star1)
	(supports instrument3 image0)
	(supports instrument3 image2)
	(calibration_target instrument3 star1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
)
(:goal (and
	(have_image planet5 image0)
	(have_image star6 image2)
	(have_image planet7 image0)
	(have_image planet8 image2)
))

)
