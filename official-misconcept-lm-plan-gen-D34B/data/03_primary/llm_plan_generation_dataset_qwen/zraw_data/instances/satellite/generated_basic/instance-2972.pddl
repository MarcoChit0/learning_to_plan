(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
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
	instrument18 - instrument
	infrared1 - mode
	infrared0 - mode
	infrared2 - mode
	thermograph3 - mode
	star0 - direction
	groundstation1 - direction
	star3 - direction
	groundstation2 - direction
	star4 - direction
	star5 - direction
)
(:init
	(supports instrument0 infrared0)
	(supports instrument0 thermograph3)
	(supports instrument0 infrared2)
	(calibration_target instrument0 star3)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument1 infrared1)
	(supports instrument1 infrared2)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 infrared1)
	(calibration_target instrument2 star3)
	(supports instrument3 infrared2)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 infrared0)
	(supports instrument4 infrared2)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 infrared1)
	(supports instrument5 infrared2)
	(calibration_target instrument5 star3)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument6 infrared2)
	(supports instrument6 infrared0)
	(calibration_target instrument6 star3)
	(supports instrument7 thermograph3)
	(supports instrument7 infrared0)
	(calibration_target instrument7 star3)
	(supports instrument8 infrared1)
	(supports instrument8 thermograph3)
	(supports instrument8 infrared2)
	(calibration_target instrument8 star3)
	(supports instrument9 infrared0)
	(supports instrument9 thermograph3)
	(supports instrument9 infrared2)
	(calibration_target instrument9 star3)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
	(supports instrument10 infrared0)
	(supports instrument10 infrared1)
	(supports instrument10 thermograph3)
	(calibration_target instrument10 groundstation2)
	(supports instrument11 infrared2)
	(supports instrument11 infrared0)
	(calibration_target instrument11 groundstation1)
	(supports instrument12 infrared0)
	(supports instrument12 infrared1)
	(supports instrument12 infrared2)
	(calibration_target instrument12 groundstation1)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star5)
	(supports instrument13 thermograph3)
	(supports instrument13 infrared2)
	(calibration_target instrument13 groundstation1)
	(supports instrument14 thermograph3)
	(calibration_target instrument14 groundstation2)
	(supports instrument15 thermograph3)
	(calibration_target instrument15 star3)
	(supports instrument16 infrared0)
	(calibration_target instrument16 star3)
	(supports instrument17 thermograph3)
	(supports instrument17 infrared0)
	(calibration_target instrument17 groundstation2)
	(supports instrument18 infrared2)
	(supports instrument18 thermograph3)
	(calibration_target instrument18 groundstation2)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star5)
)
(:goal (and
	(pointing satellite3 star5)
	(have_image star4 thermograph3)
	(have_image star5 thermograph3)
))

)
