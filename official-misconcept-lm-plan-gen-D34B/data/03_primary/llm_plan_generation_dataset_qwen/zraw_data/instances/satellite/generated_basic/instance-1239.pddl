(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
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
	infrared2 - mode
	infrared1 - mode
	spectrograph0 - mode
	spectrograph3 - mode
	star2 - direction
	groundstation3 - direction
	star4 - direction
	groundstation1 - direction
	star0 - direction
	planet5 - direction
	star6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 infrared2)
	(supports instrument0 spectrograph0)
	(supports instrument0 spectrograph3)
	(calibration_target instrument0 groundstation3)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument1 spectrograph3)
	(supports instrument1 infrared1)
	(supports instrument1 infrared2)
	(calibration_target instrument1 star0)
	(supports instrument2 infrared1)
	(supports instrument2 spectrograph3)
	(calibration_target instrument2 star2)
	(supports instrument3 infrared2)
	(calibration_target instrument3 star4)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star6)
	(supports instrument4 infrared2)
	(supports instrument4 infrared1)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 spectrograph3)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 groundstation3)
	(supports instrument6 infrared2)
	(supports instrument6 spectrograph3)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 star2)
	(supports instrument7 spectrograph3)
	(supports instrument7 infrared1)
	(calibration_target instrument7 groundstation3)
	(supports instrument8 infrared2)
	(calibration_target instrument8 star4)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 groundstation1)
	(supports instrument10 infrared1)
	(calibration_target instrument10 star0)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon8)
)
(:goal (and
	(pointing satellite1 phenomenon8)
	(pointing satellite2 star6)
	(have_image planet5 infrared2)
	(have_image star6 spectrograph3)
	(have_image phenomenon7 spectrograph3)
	(have_image phenomenon8 spectrograph3)
))

)
