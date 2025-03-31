(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite1 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	satellite4 - satellite
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	thermograph3 - mode
	thermograph2 - mode
	thermograph0 - mode
	infrared1 - mode
	star0 - direction
	groundstation1 - direction
	groundstation2 - direction
	groundstation3 - direction
	star4 - direction
	star5 - direction
	star6 - direction
)
(:init
	(supports instrument0 thermograph3)
	(calibration_target instrument0 star0)
	(supports instrument1 thermograph3)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation3)
	(supports instrument2 infrared1)
	(supports instrument2 thermograph3)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 infrared1)
	(calibration_target instrument3 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument4 thermograph2)
	(supports instrument4 infrared1)
	(calibration_target instrument4 groundstation2)
	(supports instrument5 infrared1)
	(supports instrument5 thermograph2)
	(supports instrument5 thermograph3)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 thermograph2)
	(calibration_target instrument6 star0)
	(supports instrument7 infrared1)
	(supports instrument7 thermograph2)
	(calibration_target instrument7 groundstation3)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star6)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 thermograph0)
	(supports instrument9 thermograph3)
	(supports instrument9 infrared1)
	(calibration_target instrument9 groundstation1)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star4)
	(supports instrument10 infrared1)
	(supports instrument10 thermograph2)
	(calibration_target instrument10 star0)
	(supports instrument11 thermograph3)
	(supports instrument11 thermograph2)
	(calibration_target instrument11 groundstation3)
	(supports instrument12 thermograph2)
	(calibration_target instrument12 star0)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation2)
	(supports instrument13 thermograph0)
	(supports instrument13 thermograph2)
	(supports instrument13 thermograph3)
	(calibration_target instrument13 groundstation3)
	(supports instrument14 infrared1)
	(calibration_target instrument14 star0)
	(supports instrument15 thermograph3)
	(supports instrument15 thermograph0)
	(calibration_target instrument15 groundstation1)
	(supports instrument16 thermograph0)
	(supports instrument16 thermograph2)
	(calibration_target instrument16 groundstation2)
	(supports instrument17 thermograph2)
	(supports instrument17 infrared1)
	(supports instrument17 thermograph3)
	(calibration_target instrument17 groundstation3)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star5)
)
(:goal (and
	(pointing satellite4 star5)
	(have_image star4 infrared1)
	(have_image star5 thermograph2)
	(have_image star6 thermograph3)
))

)
