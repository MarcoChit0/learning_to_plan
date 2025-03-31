(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite1 - satellite
	instrument6 - instrument
	instrument7 - instrument
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	satellite3 - satellite
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	satellite4 - satellite
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	instrument20 - instrument
	satellite5 - satellite
	instrument21 - instrument
	instrument22 - instrument
	instrument23 - instrument
	thermograph0 - mode
	groundstation4 - direction
	star1 - direction
	star0 - direction
	star3 - direction
	groundstation2 - direction
	phenomenon5 - direction
	star6 - direction
	star7 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star1)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star0)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star0)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation4)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 groundstation4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 star3)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star0)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation4)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 star0)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 star1)
	(supports instrument10 thermograph0)
	(calibration_target instrument10 groundstation4)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 star3)
	(supports instrument12 thermograph0)
	(calibration_target instrument12 star0)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon5)
	(supports instrument13 thermograph0)
	(calibration_target instrument13 star1)
	(supports instrument14 thermograph0)
	(calibration_target instrument14 star1)
	(supports instrument15 thermograph0)
	(calibration_target instrument15 star1)
	(supports instrument16 thermograph0)
	(calibration_target instrument16 groundstation2)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(on_board instrument16 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon5)
	(supports instrument17 thermograph0)
	(calibration_target instrument17 groundstation2)
	(supports instrument18 thermograph0)
	(calibration_target instrument18 star1)
	(supports instrument19 thermograph0)
	(calibration_target instrument19 star3)
	(supports instrument20 thermograph0)
	(calibration_target instrument20 star3)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(on_board instrument19 satellite4)
	(on_board instrument20 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star7)
	(supports instrument21 thermograph0)
	(calibration_target instrument21 star0)
	(supports instrument22 thermograph0)
	(calibration_target instrument22 star3)
	(supports instrument23 thermograph0)
	(calibration_target instrument23 groundstation2)
	(on_board instrument21 satellite5)
	(on_board instrument22 satellite5)
	(on_board instrument23 satellite5)
	(power_avail satellite5)
	(pointing satellite5 groundstation2)
)
(:goal (and
	(pointing satellite0 phenomenon5)
	(pointing satellite1 star1)
	(pointing satellite5 groundstation4)
	(have_image phenomenon5 thermograph0)
	(have_image star6 thermograph0)
	(have_image star7 thermograph0)
))

)
