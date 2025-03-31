(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	satellite3 - satellite
	instrument13 - instrument
	instrument14 - instrument
	satellite4 - satellite
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	instrument20 - instrument
	thermograph0 - mode
	infrared3 - mode
	infrared1 - mode
	infrared2 - mode
	groundstation3 - direction
	star1 - direction
	groundstation0 - direction
	star2 - direction
	phenomenon4 - direction
	planet5 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 infrared2)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star2)
	(supports instrument1 thermograph0)
	(supports instrument1 infrared2)
	(calibration_target instrument1 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument2 infrared1)
	(supports instrument2 infrared2)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star1)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 infrared3)
	(supports instrument4 infrared1)
	(supports instrument4 infrared2)
	(calibration_target instrument4 groundstation3)
	(supports instrument5 infrared1)
	(calibration_target instrument5 groundstation3)
	(supports instrument6 thermograph0)
	(supports instrument6 infrared3)
	(calibration_target instrument6 star1)
	(supports instrument7 infrared1)
	(supports instrument7 infrared2)
	(calibration_target instrument7 star2)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon4)
	(supports instrument8 infrared2)
	(supports instrument8 infrared3)
	(calibration_target instrument8 star1)
	(supports instrument9 infrared2)
	(supports instrument9 infrared3)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 star1)
	(supports instrument10 thermograph0)
	(supports instrument10 infrared1)
	(supports instrument10 infrared3)
	(calibration_target instrument10 star2)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 groundstation3)
	(supports instrument12 infrared3)
	(calibration_target instrument12 groundstation0)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon4)
	(supports instrument13 infrared3)
	(supports instrument13 thermograph0)
	(calibration_target instrument13 groundstation3)
	(supports instrument14 infrared1)
	(supports instrument14 infrared2)
	(calibration_target instrument14 groundstation0)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet5)
	(supports instrument15 thermograph0)
	(supports instrument15 infrared3)
	(calibration_target instrument15 star1)
	(supports instrument16 infrared3)
	(calibration_target instrument16 star1)
	(supports instrument17 infrared1)
	(calibration_target instrument17 star2)
	(supports instrument18 infrared1)
	(calibration_target instrument18 star1)
	(supports instrument19 infrared2)
	(calibration_target instrument19 groundstation0)
	(supports instrument20 infrared1)
	(supports instrument20 infrared2)
	(supports instrument20 infrared3)
	(calibration_target instrument20 star2)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(on_board instrument19 satellite4)
	(on_board instrument20 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star2)
)
(:goal (and
	(pointing satellite4 star1)
	(have_image phenomenon4 infrared2)
	(have_image planet5 infrared2)
))

)
