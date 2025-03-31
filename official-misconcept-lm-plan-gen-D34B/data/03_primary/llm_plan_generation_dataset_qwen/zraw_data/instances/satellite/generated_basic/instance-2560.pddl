(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	thermograph1 - mode
	image2 - mode
	image0 - mode
	star1 - direction
	groundstation4 - direction
	star3 - direction
	star0 - direction
	groundstation2 - direction
	star5 - direction
)
(:init
	(supports instrument0 image2)
	(supports instrument0 image0)
	(calibration_target instrument0 star3)
	(supports instrument1 image0)
	(supports instrument1 image2)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument2 image2)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 groundstation2)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation4)
)
(:goal (and
	(pointing satellite1 star0)
	(have_image star5 thermograph1)
))

)
