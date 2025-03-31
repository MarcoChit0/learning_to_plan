(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	instrument4 - instrument
	thermograph0 - mode
	image1 - mode
	star0 - direction
	star3 - direction
	star2 - direction
	star1 - direction
	planet4 - direction
	phenomenon5 - direction
	star6 - direction
	star7 - direction
	phenomenon8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 image1)
	(calibration_target instrument0 star2)
	(supports instrument1 thermograph0)
	(supports instrument1 image1)
	(calibration_target instrument1 star0)
	(supports instrument2 thermograph0)
	(supports instrument2 image1)
	(calibration_target instrument2 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star7)
	(supports instrument3 thermograph0)
	(supports instrument3 image1)
	(calibration_target instrument3 star2)
	(supports instrument4 image1)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 star1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
)
(:goal (and
	(have_image planet4 thermograph0)
	(have_image phenomenon5 thermograph0)
	(have_image star6 thermograph0)
	(have_image star7 image1)
	(have_image phenomenon8 thermograph0)
	(have_image planet9 image1)
))

)
