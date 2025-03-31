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
	satellite2 - satellite
	instrument10 - instrument
	satellite3 - satellite
	instrument11 - instrument
	instrument12 - instrument
	image0 - mode
	spectrograph3 - mode
	spectrograph2 - mode
	spectrograph1 - mode
	groundstation4 - direction
	star0 - direction
	star3 - direction
	star2 - direction
	groundstation1 - direction
	planet5 - direction
	planet6 - direction
	star7 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star3)
	(supports instrument1 spectrograph1)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 image0)
	(supports instrument2 spectrograph3)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 star0)
	(supports instrument3 spectrograph3)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 star3)
	(supports instrument4 spectrograph3)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet6)
	(supports instrument5 spectrograph2)
	(supports instrument5 spectrograph3)
	(calibration_target instrument5 star2)
	(supports instrument6 image0)
	(supports instrument6 spectrograph2)
	(calibration_target instrument6 star3)
	(supports instrument7 spectrograph1)
	(supports instrument7 spectrograph2)
	(calibration_target instrument7 star2)
	(supports instrument8 spectrograph2)
	(supports instrument8 spectrograph1)
	(calibration_target instrument8 star0)
	(supports instrument9 spectrograph3)
	(supports instrument9 spectrograph1)
	(calibration_target instrument9 star3)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star7)
	(supports instrument10 image0)
	(supports instrument10 spectrograph3)
	(calibration_target instrument10 star2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet5)
	(supports instrument11 spectrograph1)
	(calibration_target instrument11 groundstation1)
	(supports instrument12 spectrograph3)
	(supports instrument12 image0)
	(calibration_target instrument12 groundstation1)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star3)
)
(:goal (and
	(have_image planet5 image0)
	(have_image planet6 spectrograph3)
	(have_image star7 image0)
))

)
