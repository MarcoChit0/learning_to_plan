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
	instrument8 - instrument
	instrument9 - instrument
	spectrograph0 - mode
	spectrograph1 - mode
	groundstation5 - direction
	star4 - direction
	groundstation2 - direction
	groundstation3 - direction
	groundstation0 - direction
	groundstation1 - direction
	star6 - direction
	star7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation0)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 groundstation0)
	(calibration_target instrument1 groundstation5)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation2)
	(calibration_target instrument2 star4)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 groundstation2)
	(calibration_target instrument3 groundstation5)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet8)
	(supports instrument5 spectrograph0)
	(supports instrument5 spectrograph1)
	(calibration_target instrument5 groundstation2)
	(calibration_target instrument5 star4)
	(supports instrument6 spectrograph1)
	(calibration_target instrument6 groundstation2)
	(calibration_target instrument6 groundstation5)
	(supports instrument7 spectrograph1)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 star4)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 spectrograph1)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 groundstation3)
	(calibration_target instrument8 groundstation2)
	(supports instrument9 spectrograph1)
	(calibration_target instrument9 groundstation1)
	(calibration_target instrument9 groundstation0)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star7)
)
(:goal (and
	(pointing satellite0 groundstation2)
	(pointing satellite1 groundstation5)
	(have_image star6 spectrograph1)
	(have_image star7 spectrograph1)
	(have_image planet8 spectrograph0)
))

)
