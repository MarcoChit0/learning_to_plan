(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	satellite3 - satellite
	instrument6 - instrument
	satellite4 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	infrared1 - mode
	spectrograph2 - mode
	spectrograph0 - mode
	star0 - direction
	star1 - direction
	groundstation3 - direction
	star2 - direction
	star4 - direction
	planet5 - direction
	star6 - direction
	star7 - direction
	planet8 - direction
	star9 - direction
)
(:init
	(supports instrument0 infrared1)
	(calibration_target instrument0 star1)
	(supports instrument1 spectrograph0)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 star1)
	(supports instrument2 spectrograph2)
	(supports instrument2 infrared1)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument3 spectrograph0)
	(supports instrument3 spectrograph2)
	(supports instrument3 infrared1)
	(calibration_target instrument3 star2)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet5)
	(supports instrument4 spectrograph0)
	(supports instrument4 spectrograph2)
	(supports instrument4 infrared1)
	(calibration_target instrument4 star1)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 star1)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet5)
	(supports instrument6 spectrograph2)
	(calibration_target instrument6 star2)
	(on_board instrument6 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star9)
	(supports instrument7 spectrograph0)
	(supports instrument7 spectrograph2)
	(calibration_target instrument7 star1)
	(supports instrument8 spectrograph2)
	(calibration_target instrument8 star2)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 groundstation3)
	(supports instrument10 infrared1)
	(supports instrument10 spectrograph0)
	(supports instrument10 spectrograph2)
	(calibration_target instrument10 star2)
	(supports instrument11 spectrograph0)
	(supports instrument11 spectrograph2)
	(calibration_target instrument11 star2)
	(on_board instrument7 satellite4)
	(on_board instrument8 satellite4)
	(on_board instrument9 satellite4)
	(on_board instrument10 satellite4)
	(on_board instrument11 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star9)
)
(:goal (and
	(pointing satellite3 star4)
	(have_image star4 spectrograph0)
	(have_image planet5 spectrograph2)
	(have_image star6 spectrograph0)
	(have_image star7 infrared1)
	(have_image planet8 infrared1)
	(have_image star9 infrared1)
))

)
