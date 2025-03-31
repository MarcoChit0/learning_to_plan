(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	satellite4 - satellite
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	image1 - mode
	thermograph0 - mode
	groundstation2 - direction
	star0 - direction
	groundstation1 - direction
	star3 - direction
	planet4 - direction
)
(:init
	(supports instrument0 thermograph0)
	(supports instrument0 image1)
	(calibration_target instrument0 star0)
	(supports instrument1 thermograph0)
	(supports instrument1 image1)
	(calibration_target instrument1 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet4)
	(supports instrument2 thermograph0)
	(supports instrument2 image1)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 image1)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 groundstation2)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument4 image1)
	(calibration_target instrument4 star0)
	(supports instrument5 image1)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 image1)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 image1)
	(calibration_target instrument7 groundstation2)
	(supports instrument8 image1)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 image1)
	(calibration_target instrument9 star0)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
	(supports instrument10 image1)
	(supports instrument10 thermograph0)
	(calibration_target instrument10 groundstation2)
	(supports instrument11 thermograph0)
	(supports instrument11 image1)
	(calibration_target instrument11 groundstation1)
	(supports instrument12 image1)
	(calibration_target instrument12 star0)
	(supports instrument13 thermograph0)
	(calibration_target instrument13 star0)
	(supports instrument14 image1)
	(supports instrument14 thermograph0)
	(calibration_target instrument14 groundstation1)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation2)
	(supports instrument15 image1)
	(supports instrument15 thermograph0)
	(calibration_target instrument15 groundstation2)
	(supports instrument16 image1)
	(supports instrument16 thermograph0)
	(calibration_target instrument16 groundstation1)
	(supports instrument17 image1)
	(calibration_target instrument17 groundstation1)
	(supports instrument18 image1)
	(supports instrument18 thermograph0)
	(calibration_target instrument18 star0)
	(supports instrument19 thermograph0)
	(supports instrument19 image1)
	(calibration_target instrument19 groundstation1)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(on_board instrument19 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star0)
)
(:goal (and
	(pointing satellite2 star0)
	(have_image star3 thermograph0)
	(have_image planet4 image1)
))

)
