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
	image1 - mode
	infrared0 - mode
	star2 - direction
	star0 - direction
	groundstation3 - direction
	star4 - direction
	groundstation1 - direction
	phenomenon5 - direction
)
(:init
	(supports instrument0 infrared0)
	(supports instrument0 image1)
	(calibration_target instrument0 star4)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 infrared0)
	(supports instrument2 image1)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 image1)
	(calibration_target instrument3 star2)
	(supports instrument4 infrared0)
	(supports instrument4 image1)
	(calibration_target instrument4 groundstation3)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
	(supports instrument5 infrared0)
	(calibration_target instrument5 star0)
	(supports instrument6 infrared0)
	(supports instrument6 image1)
	(calibration_target instrument6 groundstation3)
	(supports instrument7 infrared0)
	(supports instrument7 image1)
	(calibration_target instrument7 groundstation1)
	(supports instrument8 infrared0)
	(calibration_target instrument8 star4)
	(supports instrument9 image1)
	(supports instrument9 infrared0)
	(calibration_target instrument9 groundstation1)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon5)
)
(:goal (and
	(pointing satellite0 groundstation1)
	(pointing satellite1 star2)
	(pointing satellite2 groundstation1)
	(have_image phenomenon5 image1)
))

)
