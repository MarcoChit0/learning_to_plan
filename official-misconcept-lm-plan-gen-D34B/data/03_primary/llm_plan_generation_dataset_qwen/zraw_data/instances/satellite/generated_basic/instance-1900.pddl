(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	satellite2 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	spectrograph0 - mode
	star0 - direction
	star4 - direction
	star1 - direction
	groundstation3 - direction
	groundstation2 - direction
	star5 - direction
	planet6 - direction
	planet7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation3)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star4)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star5)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 star1)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 star4)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 star1)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 groundstation3)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 groundstation2)
	(on_board instrument8 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star4)
)
(:goal (and
	(pointing satellite1 planet7)
	(have_image star5 spectrograph0)
	(have_image planet6 spectrograph0)
	(have_image planet7 spectrograph0)
	(have_image phenomenon8 spectrograph0)
))

)
