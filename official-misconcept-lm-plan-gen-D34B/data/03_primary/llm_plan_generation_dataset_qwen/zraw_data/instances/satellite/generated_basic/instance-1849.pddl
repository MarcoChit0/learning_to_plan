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
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	spectrograph1 - mode
	spectrograph0 - mode
	star0 - direction
	groundstation1 - direction
	groundstation2 - direction
	planet3 - direction
	phenomenon4 - direction
	planet5 - direction
	planet6 - direction
	planet7 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 spectrograph0)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 spectrograph1)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet7)
	(supports instrument5 spectrograph1)
	(calibration_target instrument5 groundstation1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon4)
	(supports instrument6 spectrograph0)
	(supports instrument6 spectrograph1)
	(calibration_target instrument6 groundstation2)
	(supports instrument7 spectrograph1)
	(calibration_target instrument7 groundstation2)
	(supports instrument8 spectrograph1)
	(calibration_target instrument8 groundstation2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon4)
)
(:goal (and
	(have_image planet3 spectrograph1)
	(have_image phenomenon4 spectrograph0)
	(have_image planet5 spectrograph1)
	(have_image planet6 spectrograph1)
	(have_image planet7 spectrograph1)
))

)
