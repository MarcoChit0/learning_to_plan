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
	satellite2 - satellite
	instrument5 - instrument
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
	image1 - mode
	image0 - mode
	infrared2 - mode
	infrared3 - mode
	star0 - direction
	star2 - direction
	star3 - direction
	groundstation1 - direction
	star4 - direction
	planet5 - direction
)
(:init
	(supports instrument0 infrared3)
	(calibration_target instrument0 star2)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet5)
	(supports instrument1 infrared3)
	(supports instrument1 image0)
	(supports instrument1 image1)
	(calibration_target instrument1 star3)
	(supports instrument2 image0)
	(supports instrument2 infrared2)
	(supports instrument2 infrared3)
	(calibration_target instrument2 star0)
	(supports instrument3 image1)
	(supports instrument3 infrared2)
	(supports instrument3 infrared3)
	(calibration_target instrument3 star3)
	(supports instrument4 infrared3)
	(calibration_target instrument4 star2)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument5 image1)
	(supports instrument5 infrared3)
	(supports instrument5 image0)
	(calibration_target instrument5 star3)
	(supports instrument6 infrared3)
	(calibration_target instrument6 star0)
	(supports instrument7 image0)
	(supports instrument7 infrared3)
	(calibration_target instrument7 star3)
	(supports instrument8 infrared3)
	(calibration_target instrument8 star2)
	(supports instrument9 image1)
	(supports instrument9 image0)
	(calibration_target instrument9 groundstation1)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star4)
	(supports instrument10 infrared3)
	(supports instrument10 image0)
	(calibration_target instrument10 star0)
	(supports instrument11 image0)
	(calibration_target instrument11 star3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation1)
	(supports instrument12 infrared3)
	(supports instrument12 infrared2)
	(supports instrument12 image1)
	(calibration_target instrument12 star0)
	(supports instrument13 infrared3)
	(supports instrument13 infrared2)
	(supports instrument13 image1)
	(calibration_target instrument13 star2)
	(supports instrument14 infrared2)
	(supports instrument14 image0)
	(supports instrument14 infrared3)
	(calibration_target instrument14 star3)
	(supports instrument15 image0)
	(supports instrument15 infrared3)
	(calibration_target instrument15 groundstation1)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet5)
)
(:goal (and
	(pointing satellite4 groundstation1)
	(have_image star4 image0)
	(have_image planet5 image0)
))

)
