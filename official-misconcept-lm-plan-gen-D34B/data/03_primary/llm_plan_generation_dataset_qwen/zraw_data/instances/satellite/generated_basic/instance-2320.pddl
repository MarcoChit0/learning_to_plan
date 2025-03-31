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
	instrument5 - instrument
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	satellite3 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	satellite4 - satellite
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	instrument20 - instrument
	instrument21 - instrument
	instrument22 - instrument
	satellite5 - satellite
	instrument23 - instrument
	instrument24 - instrument
	instrument25 - instrument
	instrument26 - instrument
	instrument27 - instrument
	instrument28 - instrument
	thermograph0 - mode
	star1 - direction
	groundstation2 - direction
	groundstation0 - direction
	planet3 - direction
	phenomenon4 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 star1)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star1)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 star1)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 star1)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 groundstation2)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 star1)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 star1)
	(supports instrument10 thermograph0)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 groundstation2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
	(supports instrument12 thermograph0)
	(calibration_target instrument12 groundstation2)
	(supports instrument13 thermograph0)
	(calibration_target instrument13 star1)
	(supports instrument14 thermograph0)
	(calibration_target instrument14 groundstation0)
	(supports instrument15 thermograph0)
	(calibration_target instrument15 groundstation2)
	(supports instrument16 thermograph0)
	(calibration_target instrument16 star1)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(on_board instrument16 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation2)
	(supports instrument17 thermograph0)
	(calibration_target instrument17 star1)
	(supports instrument18 thermograph0)
	(calibration_target instrument18 groundstation2)
	(supports instrument19 thermograph0)
	(calibration_target instrument19 groundstation0)
	(supports instrument20 thermograph0)
	(calibration_target instrument20 groundstation2)
	(supports instrument21 thermograph0)
	(calibration_target instrument21 star1)
	(supports instrument22 thermograph0)
	(calibration_target instrument22 star1)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(on_board instrument19 satellite4)
	(on_board instrument20 satellite4)
	(on_board instrument21 satellite4)
	(on_board instrument22 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet3)
	(supports instrument23 thermograph0)
	(calibration_target instrument23 groundstation0)
	(supports instrument24 thermograph0)
	(calibration_target instrument24 groundstation2)
	(supports instrument25 thermograph0)
	(calibration_target instrument25 groundstation2)
	(supports instrument26 thermograph0)
	(calibration_target instrument26 groundstation0)
	(supports instrument27 thermograph0)
	(calibration_target instrument27 groundstation0)
	(supports instrument28 thermograph0)
	(calibration_target instrument28 groundstation0)
	(on_board instrument23 satellite5)
	(on_board instrument24 satellite5)
	(on_board instrument25 satellite5)
	(on_board instrument26 satellite5)
	(on_board instrument27 satellite5)
	(on_board instrument28 satellite5)
	(power_avail satellite5)
	(pointing satellite5 groundstation0)
)
(:goal (and
	(pointing satellite0 planet3)
	(pointing satellite5 groundstation0)
	(have_image planet3 thermograph0)
	(have_image phenomenon4 thermograph0)
))

)
