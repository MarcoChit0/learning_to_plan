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
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	satellite4 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	infrared0 - mode
	infrared1 - mode
	thermograph2 - mode
	image3 - mode
	star0 - direction
	star2 - direction
	groundstation1 - direction
	star4 - direction
	star3 - direction
	star5 - direction
	star6 - direction
	star7 - direction
	planet8 - direction
	planet9 - direction
	star10 - direction
)
(:init
	(supports instrument0 thermograph2)
	(supports instrument0 infrared1)
	(calibration_target instrument0 star0)
	(supports instrument1 thermograph2)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 infrared1)
	(supports instrument2 image3)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument3 infrared0)
	(supports instrument3 infrared1)
	(supports instrument3 image3)
	(calibration_target instrument3 star0)
	(supports instrument4 image3)
	(calibration_target instrument4 star4)
	(supports instrument5 infrared1)
	(supports instrument5 thermograph2)
	(supports instrument5 infrared0)
	(calibration_target instrument5 star2)
	(supports instrument6 infrared1)
	(supports instrument6 image3)
	(calibration_target instrument6 groundstation1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument7 infrared0)
	(supports instrument7 thermograph2)
	(supports instrument7 image3)
	(calibration_target instrument7 star3)
	(supports instrument8 infrared1)
	(supports instrument8 infrared0)
	(supports instrument8 image3)
	(calibration_target instrument8 star4)
	(supports instrument9 infrared0)
	(calibration_target instrument9 star3)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star4)
	(supports instrument10 infrared1)
	(supports instrument10 infrared0)
	(supports instrument10 thermograph2)
	(calibration_target instrument10 star4)
	(on_board instrument10 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star5)
	(supports instrument11 infrared0)
	(supports instrument11 infrared1)
	(supports instrument11 image3)
	(calibration_target instrument11 star4)
	(supports instrument12 infrared1)
	(supports instrument12 infrared0)
	(calibration_target instrument12 star3)
	(supports instrument13 infrared1)
	(supports instrument13 image3)
	(calibration_target instrument13 star4)
	(supports instrument14 infrared1)
	(supports instrument14 infrared0)
	(calibration_target instrument14 star3)
	(on_board instrument11 satellite4)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star0)
)
(:goal (and
	(pointing satellite0 star10)
	(pointing satellite1 star4)
	(pointing satellite3 planet8)
	(have_image star5 thermograph2)
	(have_image star6 infrared1)
	(have_image star7 image3)
	(have_image planet8 infrared0)
	(have_image planet9 infrared1)
	(have_image star10 image3)
))

)
