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
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite3 - satellite
	instrument11 - instrument
	satellite4 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	thermograph1 - mode
	infrared2 - mode
	image0 - mode
	groundstation0 - direction
	groundstation3 - direction
	star2 - direction
	star1 - direction
	star4 - direction
	phenomenon5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 image0)
	(calibration_target instrument0 star1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon5)
	(supports instrument1 thermograph1)
	(supports instrument1 image0)
	(supports instrument1 infrared2)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 infrared2)
	(supports instrument2 image0)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 star1)
	(supports instrument3 infrared2)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 image0)
	(calibration_target instrument4 star2)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 image0)
	(supports instrument6 infrared2)
	(calibration_target instrument6 star2)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 groundstation3)
	(supports instrument8 thermograph1)
	(supports instrument8 image0)
	(calibration_target instrument8 groundstation3)
	(supports instrument9 infrared2)
	(supports instrument9 image0)
	(supports instrument9 thermograph1)
	(calibration_target instrument9 groundstation3)
	(supports instrument10 thermograph1)
	(supports instrument10 infrared2)
	(calibration_target instrument10 groundstation3)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
	(supports instrument11 image0)
	(supports instrument11 infrared2)
	(supports instrument11 thermograph1)
	(calibration_target instrument11 star2)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star4)
	(supports instrument12 image0)
	(calibration_target instrument12 groundstation3)
	(supports instrument13 infrared2)
	(supports instrument13 image0)
	(supports instrument13 thermograph1)
	(calibration_target instrument13 groundstation3)
	(supports instrument14 thermograph1)
	(supports instrument14 infrared2)
	(supports instrument14 image0)
	(calibration_target instrument14 star2)
	(supports instrument15 thermograph1)
	(calibration_target instrument15 star2)
	(supports instrument16 image0)
	(supports instrument16 infrared2)
	(calibration_target instrument16 star1)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation0)
)
(:goal (and
	(pointing satellite0 star1)
	(pointing satellite4 phenomenon5)
	(have_image star4 thermograph1)
	(have_image phenomenon5 thermograph1)
	(have_image phenomenon6 image0)
))

)
