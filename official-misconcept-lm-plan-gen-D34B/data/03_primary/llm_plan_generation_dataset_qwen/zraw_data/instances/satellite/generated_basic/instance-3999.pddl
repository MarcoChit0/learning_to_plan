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
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	satellite4 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	thermograph0 - mode
	thermograph3 - mode
	infrared1 - mode
	infrared2 - mode
	groundstation3 - direction
	star4 - direction
	groundstation1 - direction
	groundstation2 - direction
	groundstation5 - direction
	groundstation0 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 infrared1)
	(calibration_target instrument0 groundstation5)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 infrared1)
	(supports instrument2 thermograph3)
	(supports instrument2 infrared2)
	(calibration_target instrument2 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 groundstation3)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 thermograph3)
	(supports instrument4 thermograph0)
	(supports instrument4 infrared1)
	(calibration_target instrument4 star4)
	(calibration_target instrument4 groundstation3)
	(supports instrument5 thermograph0)
	(supports instrument5 infrared2)
	(supports instrument5 thermograph3)
	(calibration_target instrument5 groundstation0)
	(calibration_target instrument5 groundstation2)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon6)
	(supports instrument6 thermograph3)
	(supports instrument6 thermograph0)
	(supports instrument6 infrared2)
	(calibration_target instrument6 groundstation2)
	(calibration_target instrument6 groundstation3)
	(supports instrument7 infrared2)
	(supports instrument7 thermograph3)
	(calibration_target instrument7 groundstation5)
	(supports instrument8 infrared2)
	(calibration_target instrument8 groundstation3)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 infrared2)
	(supports instrument9 thermograph0)
	(supports instrument9 infrared1)
	(calibration_target instrument9 star4)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation3)
	(supports instrument10 thermograph0)
	(supports instrument10 thermograph3)
	(calibration_target instrument10 groundstation1)
	(supports instrument11 infrared2)
	(calibration_target instrument11 star4)
	(calibration_target instrument11 groundstation0)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star4)
	(supports instrument12 infrared2)
	(supports instrument12 infrared1)
	(calibration_target instrument12 groundstation2)
	(supports instrument13 infrared1)
	(supports instrument13 infrared2)
	(supports instrument13 thermograph0)
	(calibration_target instrument13 groundstation0)
	(calibration_target instrument13 groundstation1)
	(supports instrument14 infrared1)
	(supports instrument14 thermograph3)
	(calibration_target instrument14 groundstation1)
	(calibration_target instrument14 groundstation0)
	(supports instrument15 infrared2)
	(calibration_target instrument15 groundstation5)
	(calibration_target instrument15 groundstation2)
	(supports instrument16 thermograph0)
	(calibration_target instrument16 groundstation0)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation5)
)
(:goal (and
	(pointing satellite1 groundstation0)
	(pointing satellite4 groundstation5)
	(have_image phenomenon6 thermograph3)
))

)
