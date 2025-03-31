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
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	satellite4 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	infrared2 - mode
	image0 - mode
	thermograph3 - mode
	thermograph1 - mode
	groundstation3 - direction
	groundstation2 - direction
	groundstation1 - direction
	star5 - direction
	star0 - direction
	groundstation4 - direction
	star6 - direction
	star7 - direction
	phenomenon8 - direction
	phenomenon9 - direction
)
(:init
	(supports instrument0 thermograph3)
	(supports instrument0 thermograph1)
	(calibration_target instrument0 star5)
	(supports instrument1 image0)
	(supports instrument1 thermograph3)
	(supports instrument1 infrared2)
	(calibration_target instrument1 groundstation3)
	(calibration_target instrument1 star5)
	(supports instrument2 thermograph3)
	(calibration_target instrument2 star0)
	(supports instrument3 thermograph3)
	(supports instrument3 infrared2)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 thermograph3)
	(calibration_target instrument4 groundstation1)
	(calibration_target instrument4 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation3)
	(supports instrument5 thermograph1)
	(supports instrument5 infrared2)
	(calibration_target instrument5 groundstation2)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star7)
	(supports instrument6 thermograph1)
	(supports instrument6 infrared2)
	(supports instrument6 thermograph3)
	(calibration_target instrument6 star0)
	(supports instrument7 thermograph3)
	(calibration_target instrument7 star5)
	(calibration_target instrument7 groundstation4)
	(supports instrument8 infrared2)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 star5)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star5)
	(supports instrument9 infrared2)
	(supports instrument9 image0)
	(calibration_target instrument9 groundstation4)
	(supports instrument10 thermograph1)
	(supports instrument10 thermograph3)
	(supports instrument10 image0)
	(calibration_target instrument10 groundstation2)
	(supports instrument11 thermograph3)
	(supports instrument11 thermograph1)
	(supports instrument11 infrared2)
	(calibration_target instrument11 groundstation1)
	(calibration_target instrument11 groundstation4)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation1)
	(supports instrument12 thermograph3)
	(supports instrument12 infrared2)
	(calibration_target instrument12 groundstation1)
	(supports instrument13 thermograph3)
	(calibration_target instrument13 star5)
	(supports instrument14 thermograph1)
	(supports instrument14 infrared2)
	(supports instrument14 thermograph3)
	(calibration_target instrument14 star5)
	(supports instrument15 thermograph1)
	(supports instrument15 thermograph3)
	(supports instrument15 image0)
	(calibration_target instrument15 star0)
	(supports instrument16 infrared2)
	(supports instrument16 image0)
	(calibration_target instrument16 groundstation4)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star6)
)
(:goal (and
	(pointing satellite1 star7)
	(pointing satellite2 phenomenon8)
	(have_image star6 infrared2)
	(have_image star7 infrared2)
	(have_image phenomenon8 image0)
	(have_image phenomenon9 thermograph3)
))

)
