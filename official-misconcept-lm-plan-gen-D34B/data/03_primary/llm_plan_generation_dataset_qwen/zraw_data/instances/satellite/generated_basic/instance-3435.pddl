(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
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
	satellite4 - satellite
	instrument13 - instrument
	satellite5 - satellite
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	thermograph1 - mode
	infrared0 - mode
	infrared2 - mode
	star1 - direction
	groundstation0 - direction
	planet2 - direction
	star3 - direction
	planet4 - direction
)
(:init
	(supports instrument0 infrared0)
	(supports instrument0 thermograph1)
	(supports instrument0 infrared2)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 thermograph1)
	(supports instrument1 infrared0)
	(supports instrument1 infrared2)
	(calibration_target instrument1 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument2 infrared2)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star1)
	(supports instrument3 infrared2)
	(calibration_target instrument3 groundstation0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument4 infrared2)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star1)
	(supports instrument5 thermograph1)
	(supports instrument5 infrared0)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 thermograph1)
	(supports instrument6 infrared0)
	(supports instrument6 infrared2)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 infrared2)
	(supports instrument7 infrared0)
	(calibration_target instrument7 groundstation0)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
	(supports instrument8 infrared2)
	(supports instrument8 infrared0)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 thermograph1)
	(supports instrument9 infrared0)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 thermograph1)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 thermograph1)
	(supports instrument11 infrared2)
	(supports instrument11 infrared0)
	(calibration_target instrument11 groundstation0)
	(supports instrument12 infrared2)
	(supports instrument12 thermograph1)
	(supports instrument12 infrared0)
	(calibration_target instrument12 star1)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet2)
	(supports instrument13 infrared2)
	(supports instrument13 infrared0)
	(supports instrument13 thermograph1)
	(calibration_target instrument13 groundstation0)
	(on_board instrument13 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation0)
	(supports instrument14 infrared0)
	(supports instrument14 infrared2)
	(calibration_target instrument14 groundstation0)
	(supports instrument15 thermograph1)
	(calibration_target instrument15 groundstation0)
	(supports instrument16 infrared2)
	(supports instrument16 thermograph1)
	(supports instrument16 infrared0)
	(calibration_target instrument16 groundstation0)
	(on_board instrument14 satellite5)
	(on_board instrument15 satellite5)
	(on_board instrument16 satellite5)
	(power_avail satellite5)
	(pointing satellite5 planet4)
)
(:goal (and
	(pointing satellite0 planet4)
	(pointing satellite1 planet4)
	(have_image planet2 thermograph1)
	(have_image star3 infrared0)
	(have_image planet4 infrared2)
))

)
