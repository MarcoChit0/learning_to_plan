(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	infrared3 - mode
	spectrograph0 - mode
	image1 - mode
	infrared2 - mode
	groundstation2 - direction
	groundstation0 - direction
	groundstation1 - direction
	star3 - direction
	star4 - direction
)
(:init
	(supports instrument0 infrared2)
	(supports instrument0 infrared3)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 infrared2)
	(calibration_target instrument1 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument2 spectrograph0)
	(supports instrument2 infrared2)
	(supports instrument2 image1)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 infrared3)
	(supports instrument3 spectrograph0)
	(supports instrument3 infrared2)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 infrared2)
	(calibration_target instrument4 groundstation2)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 groundstation0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument6 infrared3)
	(supports instrument6 spectrograph0)
	(supports instrument6 image1)
	(calibration_target instrument6 groundstation2)
	(supports instrument7 spectrograph0)
	(supports instrument7 image1)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 image1)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 spectrograph0)
	(supports instrument9 image1)
	(calibration_target instrument9 groundstation1)
	(supports instrument10 spectrograph0)
	(supports instrument10 infrared2)
	(calibration_target instrument10 groundstation1)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star4)
)
(:goal (and
	(pointing satellite1 groundstation2)
	(have_image star3 spectrograph0)
	(have_image star4 spectrograph0)
))

)
