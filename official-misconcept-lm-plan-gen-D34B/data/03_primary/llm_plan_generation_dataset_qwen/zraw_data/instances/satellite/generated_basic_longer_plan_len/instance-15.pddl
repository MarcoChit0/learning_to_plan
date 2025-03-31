(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	satellite3 - satellite
	instrument7 - instrument
	satellite4 - satellite
	instrument8 - instrument
	instrument9 - instrument
	spectrograph2 - mode
	infrared1 - mode
	spectrograph0 - mode
	spectrograph3 - mode
	star2 - direction
	groundstation4 - direction
	star0 - direction
	groundstation1 - direction
	star3 - direction
	planet5 - direction
	star6 - direction
	star7 - direction
	phenomenon8 - direction
	planet9 - direction
	planet10 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 spectrograph3)
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 star0)
	(supports instrument1 spectrograph3)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star3)
	(supports instrument2 infrared1)
	(supports instrument2 spectrograph2)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon8)
	(supports instrument3 spectrograph3)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 star0)
	(supports instrument4 infrared1)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 spectrograph3)
	(supports instrument5 spectrograph2)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 groundstation1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet5)
	(supports instrument6 spectrograph3)
	(supports instrument6 spectrograph2)
	(supports instrument6 infrared1)
	(calibration_target instrument6 star3)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
	(supports instrument7 infrared1)
	(calibration_target instrument7 star3)
	(on_board instrument7 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet5)
	(supports instrument8 spectrograph2)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 spectrograph3)
	(supports instrument9 spectrograph0)
	(supports instrument9 infrared1)
	(calibration_target instrument9 star3)
	(on_board instrument8 satellite4)
	(on_board instrument9 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star3)
)
(:goal (and
	(pointing satellite0 planet5)
	(have_image planet5 spectrograph2)
	(have_image star6 spectrograph2)
	(have_image star7 spectrograph0)
	(have_image phenomenon8 spectrograph2)
	(have_image planet9 infrared1)
	(have_image planet10 spectrograph3)
))

)
