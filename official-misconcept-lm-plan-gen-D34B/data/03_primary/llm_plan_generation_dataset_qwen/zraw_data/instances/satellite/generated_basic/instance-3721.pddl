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
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	spectrograph1 - mode
	spectrograph2 - mode
	spectrograph0 - mode
	groundstation1 - direction
	groundstation2 - direction
	star4 - direction
	groundstation0 - direction
	groundstation3 - direction
	groundstation5 - direction
	star6 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 star4)
	(supports instrument1 spectrograph2)
	(supports instrument1 spectrograph0)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 star4)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 groundstation1)
	(calibration_target instrument2 groundstation3)
	(supports instrument3 spectrograph2)
	(supports instrument3 spectrograph0)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 groundstation1)
	(calibration_target instrument3 groundstation5)
	(supports instrument4 spectrograph1)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 star4)
	(calibration_target instrument4 groundstation2)
	(supports instrument5 spectrograph1)
	(supports instrument5 spectrograph2)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star6)
	(supports instrument6 spectrograph2)
	(calibration_target instrument6 groundstation5)
	(calibration_target instrument6 groundstation3)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument7 spectrograph1)
	(supports instrument7 spectrograph0)
	(supports instrument7 spectrograph2)
	(calibration_target instrument7 groundstation2)
	(supports instrument8 spectrograph2)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 star4)
	(calibration_target instrument8 groundstation5)
	(supports instrument9 spectrograph1)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 spectrograph2)
	(calibration_target instrument10 groundstation3)
	(supports instrument11 spectrograph1)
	(supports instrument11 spectrograph2)
	(supports instrument11 spectrograph0)
	(calibration_target instrument11 groundstation5)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
)
(:goal (and
	(pointing satellite0 groundstation3)
	(have_image star6 spectrograph1)
))

)
