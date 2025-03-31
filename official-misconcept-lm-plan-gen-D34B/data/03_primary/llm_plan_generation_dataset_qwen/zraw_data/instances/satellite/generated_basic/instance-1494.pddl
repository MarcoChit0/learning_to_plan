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
	instrument8 - instrument
	satellite2 - satellite
	instrument9 - instrument
	instrument10 - instrument
	satellite3 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	satellite4 - satellite
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	satellite5 - satellite
	instrument18 - instrument
	instrument19 - instrument
	thermograph1 - mode
	thermograph0 - mode
	groundstation3 - direction
	groundstation1 - direction
	groundstation5 - direction
	groundstation0 - direction
	star4 - direction
	star2 - direction
	phenomenon6 - direction
	star7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star2)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation5)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 thermograph1)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star4)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 thermograph0)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 thermograph1)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 groundstation5)
	(calibration_target instrument5 star4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 star2)
	(supports instrument7 thermograph1)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star2)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 star2)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 groundstation5)
	(supports instrument10 thermograph0)
	(supports instrument10 thermograph1)
	(calibration_target instrument10 groundstation3)
	(calibration_target instrument10 star2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
	(supports instrument11 thermograph1)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 groundstation5)
	(calibration_target instrument11 star4)
	(supports instrument12 thermograph0)
	(supports instrument12 thermograph1)
	(calibration_target instrument12 star2)
	(calibration_target instrument12 groundstation0)
	(supports instrument13 thermograph1)
	(supports instrument13 thermograph0)
	(calibration_target instrument13 groundstation3)
	(calibration_target instrument13 groundstation0)
	(supports instrument14 thermograph0)
	(supports instrument14 thermograph1)
	(calibration_target instrument14 groundstation1)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation5)
	(supports instrument15 thermograph0)
	(calibration_target instrument15 star2)
	(calibration_target instrument15 star4)
	(supports instrument16 thermograph0)
	(supports instrument16 thermograph1)
	(calibration_target instrument16 groundstation5)
	(supports instrument17 thermograph1)
	(calibration_target instrument17 star4)
	(calibration_target instrument17 groundstation0)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(power_avail satellite4)
	(pointing satellite4 phenomenon8)
	(supports instrument18 thermograph0)
	(supports instrument18 thermograph1)
	(calibration_target instrument18 star2)
	(supports instrument19 thermograph0)
	(calibration_target instrument19 star2)
	(on_board instrument18 satellite5)
	(on_board instrument19 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star4)
)
(:goal (and
	(pointing satellite0 groundstation5)
	(pointing satellite1 phenomenon8)
	(pointing satellite2 groundstation3)
	(have_image phenomenon6 thermograph0)
	(have_image star7 thermograph1)
	(have_image phenomenon8 thermograph0)
))

)
