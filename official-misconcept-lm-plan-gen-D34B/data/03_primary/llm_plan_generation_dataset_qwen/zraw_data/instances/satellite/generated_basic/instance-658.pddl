(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	image0 - mode
	thermograph1 - mode
	star1 - direction
	star2 - direction
	star3 - direction
	star5 - direction
	groundstation0 - direction
	star4 - direction
	planet6 - direction
	star7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 image0)
	(calibration_target instrument0 star5)
	(calibration_target instrument0 groundstation0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument1 thermograph1)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 image0)
	(calibration_target instrument2 star4)
	(supports instrument3 image0)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 star4)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 star4)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star7)
)
(:goal (and
	(have_image planet6 image0)
	(have_image star7 image0)
	(have_image planet8 thermograph1)
))

)
