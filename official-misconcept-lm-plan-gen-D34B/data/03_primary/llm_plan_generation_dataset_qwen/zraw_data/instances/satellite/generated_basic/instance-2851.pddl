(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	satellite2 - satellite
	instrument2 - instrument
	instrument3 - instrument
	satellite3 - satellite
	instrument4 - instrument
	image1 - mode
	thermograph0 - mode
	groundstation2 - direction
	star4 - direction
	groundstation3 - direction
	star0 - direction
	star1 - direction
	planet5 - direction
	phenomenon6 - direction
	planet7 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star4)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet7)
	(supports instrument1 thermograph0)
	(supports instrument1 image1)
	(calibration_target instrument1 groundstation3)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet5)
	(supports instrument2 image1)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star1)
	(supports instrument3 image1)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star0)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon6)
	(supports instrument4 thermograph0)
	(supports instrument4 image1)
	(calibration_target instrument4 star1)
	(on_board instrument4 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet5)
)
(:goal (and
	(pointing satellite3 groundstation3)
	(have_image planet5 image1)
	(have_image phenomenon6 thermograph0)
	(have_image planet7 thermograph0)
))

)
