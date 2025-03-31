(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite1 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite2 - satellite
	instrument9 - instrument
	instrument10 - instrument
	spectrograph0 - mode
	spectrograph2 - mode
	infrared1 - mode
	spectrograph3 - mode
	star2 - direction
	groundstation1 - direction
	groundstation0 - direction
	phenomenon3 - direction
	planet4 - direction
)
(:init
	(supports instrument0 spectrograph3)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 infrared1)
	(supports instrument1 spectrograph0)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 spectrograph2)
	(supports instrument2 infrared1)
	(calibration_target instrument2 star2)
	(supports instrument3 infrared1)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon3)
	(supports instrument4 spectrograph2)
	(supports instrument4 spectrograph0)
	(supports instrument4 spectrograph3)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 spectrograph3)
	(supports instrument5 spectrograph0)
	(supports instrument5 infrared1)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 spectrograph3)
	(calibration_target instrument6 star2)
	(supports instrument7 infrared1)
	(calibration_target instrument7 star2)
	(supports instrument8 infrared1)
	(supports instrument8 spectrograph3)
	(supports instrument8 spectrograph2)
	(calibration_target instrument8 groundstation1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon3)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 spectrograph3)
	(supports instrument10 infrared1)
	(calibration_target instrument10 groundstation0)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
)
(:goal (and
	(pointing satellite0 groundstation0)
	(pointing satellite1 phenomenon3)
	(pointing satellite2 star2)
	(have_image phenomenon3 spectrograph3)
	(have_image planet4 spectrograph2)
))

)
