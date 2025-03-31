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
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite3 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	infrared0 - mode
	infrared3 - mode
	image2 - mode
	infrared1 - mode
	star0 - direction
	planet1 - direction
	phenomenon2 - direction
	star3 - direction
)
(:init
	(supports instrument0 infrared3)
	(supports instrument0 infrared0)
	(calibration_target instrument0 star0)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star0)
	(supports instrument2 infrared3)
	(calibration_target instrument2 star0)
	(supports instrument3 infrared1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star0)
	(supports instrument4 infrared3)
	(calibration_target instrument4 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument5 infrared1)
	(supports instrument5 image2)
	(supports instrument5 infrared3)
	(calibration_target instrument5 star0)
	(supports instrument6 infrared1)
	(calibration_target instrument6 star0)
	(supports instrument7 image2)
	(supports instrument7 infrared0)
	(supports instrument7 infrared3)
	(calibration_target instrument7 star0)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument8 image2)
	(supports instrument8 infrared0)
	(supports instrument8 infrared1)
	(calibration_target instrument8 star0)
	(supports instrument9 infrared0)
	(supports instrument9 infrared1)
	(calibration_target instrument9 star0)
	(supports instrument10 image2)
	(supports instrument10 infrared0)
	(supports instrument10 infrared1)
	(calibration_target instrument10 star0)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
	(supports instrument11 infrared3)
	(supports instrument11 infrared0)
	(calibration_target instrument11 star0)
	(supports instrument12 image2)
	(supports instrument12 infrared0)
	(supports instrument12 infrared1)
	(calibration_target instrument12 star0)
	(supports instrument13 infrared1)
	(calibration_target instrument13 star0)
	(supports instrument14 image2)
	(supports instrument14 infrared3)
	(supports instrument14 infrared1)
	(calibration_target instrument14 star0)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet1)
)
(:goal (and
	(pointing satellite0 star3)
	(have_image planet1 image2)
	(have_image phenomenon2 infrared3)
	(have_image star3 image2)
))

)
