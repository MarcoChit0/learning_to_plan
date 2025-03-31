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
	instrument7 - instrument
	instrument8 - instrument
	satellite2 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	satellite3 - satellite
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	thermograph0 - mode
	groundstation2 - direction
	groundstation4 - direction
	groundstation3 - direction
	star0 - direction
	star1 - direction
	phenomenon5 - direction
	phenomenon6 - direction
	star7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star0)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 star1)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon6)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation4)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 star0)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 groundstation3)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star0)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 groundstation2)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon5)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 star0)
	(supports instrument10 thermograph0)
	(calibration_target instrument10 star0)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 groundstation2)
	(supports instrument12 thermograph0)
	(calibration_target instrument12 groundstation4)
	(supports instrument13 thermograph0)
	(calibration_target instrument13 groundstation4)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon8)
	(supports instrument14 thermograph0)
	(calibration_target instrument14 groundstation3)
	(supports instrument15 thermograph0)
	(calibration_target instrument15 groundstation3)
	(supports instrument16 thermograph0)
	(calibration_target instrument16 star0)
	(supports instrument17 thermograph0)
	(calibration_target instrument17 star1)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(on_board instrument16 satellite3)
	(on_board instrument17 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star1)
)
(:goal (and
	(pointing satellite3 groundstation3)
	(have_image phenomenon5 thermograph0)
	(have_image phenomenon6 thermograph0)
	(have_image star7 thermograph0)
	(have_image phenomenon8 thermograph0)
))

)
