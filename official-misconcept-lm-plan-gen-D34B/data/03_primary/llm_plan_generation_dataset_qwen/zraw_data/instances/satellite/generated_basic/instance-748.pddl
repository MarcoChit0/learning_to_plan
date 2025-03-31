(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite1 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite2 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	satellite3 - satellite
	instrument15 - instrument
	infrared2 - mode
	image1 - mode
	infrared0 - mode
	star5 - direction
	star1 - direction
	star2 - direction
	groundstation4 - direction
	star3 - direction
	star0 - direction
	phenomenon6 - direction
	planet7 - direction
	phenomenon8 - direction
	star9 - direction
)
(:init
	(supports instrument0 infrared2)
	(calibration_target instrument0 star5)
	(supports instrument1 image1)
	(supports instrument1 infrared2)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star3)
	(calibration_target instrument1 star5)
	(supports instrument2 infrared2)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star5)
	(supports instrument3 infrared0)
	(supports instrument3 image1)
	(calibration_target instrument3 star2)
	(calibration_target instrument3 star0)
	(supports instrument4 infrared0)
	(supports instrument4 image1)
	(calibration_target instrument4 star5)
	(supports instrument5 infrared0)
	(supports instrument5 image1)
	(calibration_target instrument5 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument6 infrared2)
	(supports instrument6 image1)
	(supports instrument6 infrared0)
	(calibration_target instrument6 star2)
	(calibration_target instrument6 star3)
	(supports instrument7 infrared0)
	(supports instrument7 infrared2)
	(calibration_target instrument7 star2)
	(calibration_target instrument7 groundstation4)
	(supports instrument8 infrared0)
	(calibration_target instrument8 star1)
	(supports instrument9 infrared2)
	(supports instrument9 infrared0)
	(supports instrument9 image1)
	(calibration_target instrument9 star0)
	(calibration_target instrument9 star2)
	(supports instrument10 infrared2)
	(supports instrument10 image1)
	(calibration_target instrument10 star3)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(on_board instrument10 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet7)
	(supports instrument11 image1)
	(supports instrument11 infrared0)
	(calibration_target instrument11 star2)
	(calibration_target instrument11 star3)
	(supports instrument12 infrared2)
	(supports instrument12 infrared0)
	(supports instrument12 image1)
	(calibration_target instrument12 star2)
	(calibration_target instrument12 star3)
	(supports instrument13 infrared2)
	(supports instrument13 image1)
	(calibration_target instrument13 groundstation4)
	(calibration_target instrument13 star0)
	(supports instrument14 infrared0)
	(supports instrument14 infrared2)
	(supports instrument14 image1)
	(calibration_target instrument14 star3)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(on_board instrument14 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
	(supports instrument15 image1)
	(supports instrument15 infrared2)
	(calibration_target instrument15 star0)
	(on_board instrument15 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon8)
)
(:goal (and
	(pointing satellite0 star0)
	(have_image phenomenon6 infrared2)
	(have_image planet7 infrared0)
	(have_image phenomenon8 image1)
	(have_image star9 infrared2)
))

)
