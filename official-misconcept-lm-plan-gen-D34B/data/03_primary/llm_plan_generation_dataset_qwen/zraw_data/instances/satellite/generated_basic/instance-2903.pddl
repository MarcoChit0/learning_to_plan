(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite1 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite2 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	satellite3 - satellite
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	satellite4 - satellite
	instrument20 - instrument
	instrument21 - instrument
	infrared2 - mode
	infrared1 - mode
	infrared3 - mode
	thermograph0 - mode
	star0 - direction
	groundstation1 - direction
	star2 - direction
	phenomenon3 - direction
)
(:init
	(supports instrument0 infrared3)
	(supports instrument0 infrared2)
	(calibration_target instrument0 star0)
	(supports instrument1 infrared3)
	(calibration_target instrument1 star0)
	(supports instrument2 infrared3)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 infrared2)
	(supports instrument3 infrared3)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 infrared2)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument5 thermograph0)
	(supports instrument5 infrared2)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 infrared3)
	(supports instrument6 infrared2)
	(calibration_target instrument6 star0)
	(supports instrument7 infrared1)
	(calibration_target instrument7 star0)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 star0)
	(supports instrument9 infrared2)
	(calibration_target instrument9 star0)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon3)
	(supports instrument10 infrared2)
	(calibration_target instrument10 groundstation1)
	(supports instrument11 infrared2)
	(supports instrument11 thermograph0)
	(supports instrument11 infrared3)
	(calibration_target instrument11 star0)
	(supports instrument12 infrared1)
	(supports instrument12 infrared2)
	(supports instrument12 infrared3)
	(calibration_target instrument12 star0)
	(supports instrument13 infrared1)
	(calibration_target instrument13 groundstation1)
	(supports instrument14 thermograph0)
	(supports instrument14 infrared3)
	(supports instrument14 infrared1)
	(calibration_target instrument14 groundstation1)
	(supports instrument15 thermograph0)
	(supports instrument15 infrared2)
	(supports instrument15 infrared1)
	(calibration_target instrument15 star0)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(on_board instrument14 satellite2)
	(on_board instrument15 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
	(supports instrument16 infrared3)
	(supports instrument16 thermograph0)
	(calibration_target instrument16 star0)
	(supports instrument17 infrared1)
	(calibration_target instrument17 star0)
	(supports instrument18 infrared1)
	(calibration_target instrument18 star0)
	(supports instrument19 infrared3)
	(supports instrument19 infrared1)
	(supports instrument19 infrared2)
	(calibration_target instrument19 groundstation1)
	(on_board instrument16 satellite3)
	(on_board instrument17 satellite3)
	(on_board instrument18 satellite3)
	(on_board instrument19 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation1)
	(supports instrument20 infrared3)
	(supports instrument20 infrared2)
	(supports instrument20 thermograph0)
	(calibration_target instrument20 groundstation1)
	(supports instrument21 infrared3)
	(supports instrument21 thermograph0)
	(calibration_target instrument21 groundstation1)
	(on_board instrument20 satellite4)
	(on_board instrument21 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation1)
)
(:goal (and
	(pointing satellite1 groundstation1)
	(pointing satellite4 star0)
	(have_image star2 infrared2)
	(have_image phenomenon3 thermograph0)
))

)
