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
	satellite3 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite4 - satellite
	instrument9 - instrument
	satellite5 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	infrared1 - mode
	thermograph0 - mode
	star0 - direction
	star4 - direction
	star5 - direction
	star2 - direction
	groundstation3 - direction
	groundstation1 - direction
	star6 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star2)
	(calibration_target instrument0 groundstation1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument1 thermograph0)
	(supports instrument1 infrared1)
	(calibration_target instrument1 groundstation3)
	(calibration_target instrument1 star4)
	(supports instrument2 infrared1)
	(calibration_target instrument2 star5)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star2)
	(supports instrument4 infrared1)
	(calibration_target instrument4 groundstation3)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
	(supports instrument5 infrared1)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 star2)
	(supports instrument6 infrared1)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star4)
	(calibration_target instrument7 star5)
	(supports instrument8 infrared1)
	(calibration_target instrument8 star2)
	(calibration_target instrument8 star5)
	(on_board instrument5 satellite3)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star4)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 star2)
	(on_board instrument9 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star0)
	(supports instrument10 thermograph0)
	(supports instrument10 infrared1)
	(calibration_target instrument10 star4)
	(calibration_target instrument10 groundstation1)
	(supports instrument11 thermograph0)
	(supports instrument11 infrared1)
	(calibration_target instrument11 star4)
	(supports instrument12 infrared1)
	(supports instrument12 thermograph0)
	(calibration_target instrument12 star5)
	(calibration_target instrument12 star2)
	(supports instrument13 thermograph0)
	(supports instrument13 infrared1)
	(calibration_target instrument13 star2)
	(supports instrument14 infrared1)
	(supports instrument14 thermograph0)
	(calibration_target instrument14 groundstation3)
	(supports instrument15 infrared1)
	(calibration_target instrument15 groundstation1)
	(on_board instrument10 satellite5)
	(on_board instrument11 satellite5)
	(on_board instrument12 satellite5)
	(on_board instrument13 satellite5)
	(on_board instrument14 satellite5)
	(on_board instrument15 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star2)
)
(:goal (and
	(pointing satellite3 star4)
	(pointing satellite5 star2)
	(have_image star6 thermograph0)
))

)
