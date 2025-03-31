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
	instrument10 - instrument
	satellite3 - satellite
	instrument11 - instrument
	instrument12 - instrument
	spectrograph1 - mode
	spectrograph2 - mode
	spectrograph0 - mode
	groundstation0 - direction
	star1 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 groundstation0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 spectrograph1)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 spectrograph0)
	(supports instrument3 spectrograph2)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 spectrograph0)
	(supports instrument4 spectrograph2)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 groundstation0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument5 spectrograph0)
	(supports instrument5 spectrograph2)
	(supports instrument5 spectrograph1)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 spectrograph1)
	(supports instrument6 spectrograph2)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 spectrograph1)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 spectrograph0)
	(supports instrument9 spectrograph1)
	(supports instrument9 spectrograph2)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 spectrograph2)
	(calibration_target instrument10 groundstation0)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
	(supports instrument11 spectrograph1)
	(calibration_target instrument11 groundstation0)
	(supports instrument12 spectrograph2)
	(supports instrument12 spectrograph1)
	(supports instrument12 spectrograph0)
	(calibration_target instrument12 groundstation0)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation0)
)
(:goal (and
	(pointing satellite0 groundstation0)
	(have_image star1 spectrograph2)
))

)
