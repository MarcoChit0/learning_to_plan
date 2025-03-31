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
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	instrument10 - instrument
	infrared0 - mode
	infrared2 - mode
	image1 - mode
	spectrograph3 - mode
	star2 - direction
	star0 - direction
	groundstation1 - direction
	star3 - direction
	star4 - direction
	planet5 - direction
	star6 - direction
	star7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 star0)
	(supports instrument1 infrared0)
	(supports instrument1 infrared2)
	(calibration_target instrument1 star4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument2 infrared2)
	(supports instrument2 image1)
	(calibration_target instrument2 star3)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star3)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 image1)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 infrared2)
	(calibration_target instrument6 star4)
	(supports instrument7 image1)
	(supports instrument7 infrared0)
	(calibration_target instrument7 star4)
	(supports instrument8 infrared0)
	(supports instrument8 image1)
	(calibration_target instrument8 groundstation1)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
	(supports instrument9 image1)
	(supports instrument9 infrared2)
	(supports instrument9 spectrograph3)
	(calibration_target instrument9 star3)
	(supports instrument10 image1)
	(supports instrument10 infrared2)
	(supports instrument10 infrared0)
	(calibration_target instrument10 star4)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star4)
)
(:goal (and
	(have_image planet5 image1)
	(have_image star6 spectrograph3)
	(have_image star7 infrared2)
	(have_image planet8 infrared0)
))

)
