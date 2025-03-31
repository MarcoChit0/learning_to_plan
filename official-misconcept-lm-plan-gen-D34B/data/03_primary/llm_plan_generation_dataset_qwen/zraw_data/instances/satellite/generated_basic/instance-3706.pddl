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
	satellite2 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	spectrograph2 - mode
	infrared1 - mode
	spectrograph0 - mode
	star0 - direction
	star2 - direction
	groundstation1 - direction
	star4 - direction
	groundstation5 - direction
	star3 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 star2)
	(supports instrument1 spectrograph0)
	(supports instrument1 infrared1)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 star2)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 star3)
	(calibration_target instrument2 groundstation5)
	(supports instrument3 spectrograph2)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 star0)
	(calibration_target instrument3 groundstation5)
	(supports instrument4 spectrograph2)
	(calibration_target instrument4 groundstation5)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon6)
	(supports instrument5 infrared1)
	(supports instrument5 spectrograph2)
	(calibration_target instrument5 star4)
	(supports instrument6 spectrograph2)
	(calibration_target instrument6 star2)
	(supports instrument7 spectrograph2)
	(calibration_target instrument7 groundstation5)
	(supports instrument8 infrared1)
	(calibration_target instrument8 groundstation1)
	(calibration_target instrument8 star4)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 star4)
	(calibration_target instrument9 groundstation5)
	(supports instrument10 spectrograph2)
	(supports instrument10 infrared1)
	(supports instrument10 spectrograph0)
	(calibration_target instrument10 star3)
	(calibration_target instrument10 groundstation5)
	(supports instrument11 infrared1)
	(supports instrument11 spectrograph0)
	(supports instrument11 spectrograph2)
	(calibration_target instrument11 star3)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
)
(:goal (and
	(pointing satellite0 star2)
	(have_image phenomenon6 infrared1)
))

)
