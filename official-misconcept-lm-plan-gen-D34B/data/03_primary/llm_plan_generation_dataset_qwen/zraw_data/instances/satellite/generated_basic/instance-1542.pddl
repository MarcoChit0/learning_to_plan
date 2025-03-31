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
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	satellite3 - satellite
	instrument12 - instrument
	image1 - mode
	infrared2 - mode
	spectrograph0 - mode
	star0 - direction
	planet1 - direction
)
(:init
	(supports instrument0 image1)
	(calibration_target instrument0 star0)
	(supports instrument1 infrared2)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star0)
	(supports instrument2 infrared2)
	(supports instrument2 image1)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 star0)
	(supports instrument3 image1)
	(supports instrument3 spectrograph0)
	(supports instrument3 infrared2)
	(calibration_target instrument3 star0)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet1)
	(supports instrument5 image1)
	(supports instrument5 spectrograph0)
	(supports instrument5 infrared2)
	(calibration_target instrument5 star0)
	(supports instrument6 image1)
	(calibration_target instrument6 star0)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument7 spectrograph0)
	(supports instrument7 infrared2)
	(supports instrument7 image1)
	(calibration_target instrument7 star0)
	(supports instrument8 image1)
	(supports instrument8 infrared2)
	(calibration_target instrument8 star0)
	(supports instrument9 image1)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 star0)
	(supports instrument10 infrared2)
	(calibration_target instrument10 star0)
	(supports instrument11 spectrograph0)
	(supports instrument11 infrared2)
	(supports instrument11 image1)
	(calibration_target instrument11 star0)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
	(supports instrument12 infrared2)
	(supports instrument12 spectrograph0)
	(calibration_target instrument12 star0)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet1)
)
(:goal (and
	(pointing satellite0 planet1)
	(pointing satellite1 planet1)
	(have_image planet1 image1)
))

)
