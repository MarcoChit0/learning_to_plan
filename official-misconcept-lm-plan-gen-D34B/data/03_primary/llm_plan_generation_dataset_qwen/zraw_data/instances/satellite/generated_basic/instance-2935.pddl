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
	spectrograph2 - mode
	image1 - mode
	spectrograph0 - mode
	star4 - direction
	star0 - direction
	star2 - direction
	groundstation1 - direction
	groundstation3 - direction
	planet5 - direction
	planet6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 image1)
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 image1)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 star0)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 star2)
	(supports instrument3 image1)
	(supports instrument3 spectrograph2)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 image1)
	(supports instrument4 spectrograph2)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet5)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 star2)
	(supports instrument6 spectrograph2)
	(supports instrument6 spectrograph0)
	(supports instrument6 image1)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 spectrograph0)
	(supports instrument7 image1)
	(calibration_target instrument7 groundstation3)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet6)
)
(:goal (and
	(have_image planet5 spectrograph0)
	(have_image planet6 image1)
	(have_image phenomenon7 image1)
	(have_image phenomenon8 image1)
))

)
