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
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	satellite4 - satellite
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	instrument20 - instrument
	satellite5 - satellite
	instrument21 - instrument
	instrument22 - instrument
	instrument23 - instrument
	instrument24 - instrument
	thermograph0 - mode
	star0 - direction
	groundstation1 - direction
	planet2 - direction
	planet3 - direction
	planet4 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star0)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 star0)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star0)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star0)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 star0)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet4)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 groundstation1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet2)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 groundstation1)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 star0)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 groundstation1)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet3)
	(supports instrument10 thermograph0)
	(calibration_target instrument10 groundstation1)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 star0)
	(supports instrument12 thermograph0)
	(calibration_target instrument12 star0)
	(supports instrument13 thermograph0)
	(calibration_target instrument13 groundstation1)
	(supports instrument14 thermograph0)
	(calibration_target instrument14 star0)
	(supports instrument15 thermograph0)
	(calibration_target instrument15 groundstation1)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
	(supports instrument16 thermograph0)
	(calibration_target instrument16 star0)
	(supports instrument17 thermograph0)
	(calibration_target instrument17 groundstation1)
	(supports instrument18 thermograph0)
	(calibration_target instrument18 star0)
	(supports instrument19 thermograph0)
	(calibration_target instrument19 groundstation1)
	(supports instrument20 thermograph0)
	(calibration_target instrument20 star0)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(on_board instrument19 satellite4)
	(on_board instrument20 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet3)
	(supports instrument21 thermograph0)
	(calibration_target instrument21 groundstation1)
	(supports instrument22 thermograph0)
	(calibration_target instrument22 groundstation1)
	(supports instrument23 thermograph0)
	(calibration_target instrument23 groundstation1)
	(supports instrument24 thermograph0)
	(calibration_target instrument24 groundstation1)
	(on_board instrument21 satellite5)
	(on_board instrument22 satellite5)
	(on_board instrument23 satellite5)
	(on_board instrument24 satellite5)
	(power_avail satellite5)
	(pointing satellite5 planet3)
)
(:goal (and
	(pointing satellite4 star0)
	(pointing satellite5 star0)
	(have_image planet2 thermograph0)
	(have_image planet3 thermograph0)
	(have_image planet4 thermograph0)
))

)
