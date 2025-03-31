(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	thermograph0 - mode
	image1 - mode
	star1 - direction
	star3 - direction
	star0 - direction
	star2 - direction
	groundstation4 - direction
	star5 - direction
	planet6 - direction
	star7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 thermograph0)
	(supports instrument0 image1)
	(calibration_target instrument0 star0)
	(supports instrument1 image1)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument2 image1)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 groundstation4)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star7)
)
(:goal (and
	(pointing satellite1 planet6)
	(have_image star5 image1)
	(have_image planet6 thermograph0)
	(have_image star7 thermograph0)
	(have_image phenomenon8 thermograph0)
))

)
