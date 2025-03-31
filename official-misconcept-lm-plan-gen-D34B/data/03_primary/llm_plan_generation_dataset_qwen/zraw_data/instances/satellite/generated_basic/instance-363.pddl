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
	spectrograph1 - mode
	spectrograph0 - mode
	star3 - direction
	star0 - direction
	star1 - direction
	groundstation2 - direction
	planet4 - direction
	star5 - direction
	planet6 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 star3)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument1 spectrograph0)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 star0)
	(supports instrument2 spectrograph0)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 star0)
	(supports instrument3 spectrograph0)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 star1)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 groundstation2)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet4)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 spectrograph1)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 star0)
	(supports instrument7 spectrograph1)
	(calibration_target instrument7 star0)
	(supports instrument8 spectrograph0)
	(supports instrument8 spectrograph1)
	(calibration_target instrument8 star1)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 groundstation2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
)
(:goal (and
	(pointing satellite0 planet6)
	(pointing satellite1 star1)
	(have_image planet4 spectrograph0)
	(have_image star5 spectrograph0)
	(have_image planet6 spectrograph0)
))

)
