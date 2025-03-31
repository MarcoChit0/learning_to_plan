(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	satellite4 - satellite
	instrument14 - instrument
	satellite5 - satellite
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	thermograph0 - mode
	groundstation2 - direction
	star3 - direction
	groundstation0 - direction
	groundstation5 - direction
	star1 - direction
	star4 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 groundstation5)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star4)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 groundstation5)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation5)
	(calibration_target instrument4 star3)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 groundstation0)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 star4)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star3)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star4)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 groundstation5)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 groundstation5)
	(supports instrument10 thermograph0)
	(calibration_target instrument10 groundstation2)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 groundstation2)
	(supports instrument12 thermograph0)
	(calibration_target instrument12 star1)
	(supports instrument13 thermograph0)
	(calibration_target instrument13 star4)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star1)
	(supports instrument14 thermograph0)
	(calibration_target instrument14 groundstation5)
	(calibration_target instrument14 star3)
	(on_board instrument14 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star4)
	(supports instrument15 thermograph0)
	(calibration_target instrument15 star4)
	(calibration_target instrument15 star3)
	(supports instrument16 thermograph0)
	(calibration_target instrument16 groundstation0)
	(supports instrument17 thermograph0)
	(calibration_target instrument17 star1)
	(calibration_target instrument17 groundstation5)
	(supports instrument18 thermograph0)
	(calibration_target instrument18 star4)
	(on_board instrument15 satellite5)
	(on_board instrument16 satellite5)
	(on_board instrument17 satellite5)
	(on_board instrument18 satellite5)
	(power_avail satellite5)
	(pointing satellite5 groundstation2)
)
(:goal (and
	(pointing satellite0 star4)
	(pointing satellite4 star3)
	(have_image phenomenon6 thermograph0)
))

)
