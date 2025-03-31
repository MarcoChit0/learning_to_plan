(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite2 - satellite
	instrument5 - instrument
	image2 - mode
	thermograph1 - mode
	thermograph0 - mode
	star0 - direction
	star3 - direction
	star4 - direction
	groundstation5 - direction
	star2 - direction
	star1 - direction
	planet6 - direction
	star7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 thermograph1)
	(calibration_target instrument0 star2)
	(supports instrument1 thermograph1)
	(supports instrument1 thermograph0)
	(supports instrument1 image2)
	(calibration_target instrument1 groundstation5)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet8)
	(supports instrument2 image2)
	(supports instrument2 thermograph1)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 groundstation5)
	(supports instrument3 thermograph0)
	(supports instrument3 thermograph1)
	(supports instrument3 image2)
	(calibration_target instrument3 star2)
	(calibration_target instrument3 groundstation5)
	(supports instrument4 thermograph1)
	(supports instrument4 thermograph0)
	(supports instrument4 image2)
	(calibration_target instrument4 star1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star7)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 star1)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
)
(:goal (and
	(pointing satellite0 star1)
	(pointing satellite2 planet6)
	(have_image planet6 thermograph0)
	(have_image star7 thermograph1)
	(have_image planet8 image2)
))

)
