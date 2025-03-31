(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	satellite3 - satellite
	instrument7 - instrument
	instrument8 - instrument
	satellite4 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	satellite5 - satellite
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	thermograph0 - mode
	star2 - direction
	star0 - direction
	groundstation4 - direction
	star1 - direction
	groundstation3 - direction
	groundstation5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 groundstation5)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon6)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 star2)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 groundstation3)
	(calibration_target instrument2 star1)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 groundstation3)
	(calibration_target instrument3 groundstation4)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation4)
	(calibration_target instrument4 star1)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 star0)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 groundstation4)
	(calibration_target instrument6 groundstation3)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation4)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star2)
	(calibration_target instrument7 groundstation3)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 star0)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation4)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 star1)
	(supports instrument10 thermograph0)
	(calibration_target instrument10 star0)
	(calibration_target instrument10 groundstation5)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 star1)
	(calibration_target instrument11 star0)
	(supports instrument12 thermograph0)
	(calibration_target instrument12 star2)
	(supports instrument13 thermograph0)
	(calibration_target instrument13 star0)
	(calibration_target instrument13 star2)
	(on_board instrument9 satellite4)
	(on_board instrument10 satellite4)
	(on_board instrument11 satellite4)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation5)
	(supports instrument14 thermograph0)
	(calibration_target instrument14 star1)
	(calibration_target instrument14 star2)
	(supports instrument15 thermograph0)
	(calibration_target instrument15 star0)
	(calibration_target instrument15 star2)
	(supports instrument16 thermograph0)
	(calibration_target instrument16 groundstation4)
	(supports instrument17 thermograph0)
	(calibration_target instrument17 groundstation3)
	(calibration_target instrument17 star1)
	(supports instrument18 thermograph0)
	(calibration_target instrument18 groundstation5)
	(on_board instrument14 satellite5)
	(on_board instrument15 satellite5)
	(on_board instrument16 satellite5)
	(on_board instrument17 satellite5)
	(on_board instrument18 satellite5)
	(power_avail satellite5)
	(pointing satellite5 groundstation3)
)
(:goal (and
	(pointing satellite0 groundstation3)
	(pointing satellite1 groundstation5)
	(have_image phenomenon6 thermograph0)
))

)
