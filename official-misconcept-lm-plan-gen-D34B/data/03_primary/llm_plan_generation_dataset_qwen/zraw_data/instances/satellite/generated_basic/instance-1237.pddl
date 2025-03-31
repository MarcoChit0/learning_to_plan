(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	infrared1 - mode
	spectrograph2 - mode
	spectrograph0 - mode
	infrared3 - mode
	groundstation0 - direction
	groundstation3 - direction
	groundstation2 - direction
	star4 - direction
	star1 - direction
	phenomenon5 - direction
	planet6 - direction
	star7 - direction
	star8 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 spectrograph0)
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 infrared3)
	(calibration_target instrument1 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument2 infrared3)
	(calibration_target instrument2 star4)
	(supports instrument3 spectrograph0)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 spectrograph2)
	(calibration_target instrument4 groundstation3)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star7)
	(supports instrument5 spectrograph2)
	(supports instrument5 infrared3)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 groundstation3)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 star1)
	(supports instrument7 spectrograph0)
	(supports instrument7 infrared3)
	(supports instrument7 spectrograph2)
	(calibration_target instrument7 star1)
	(supports instrument8 spectrograph2)
	(supports instrument8 infrared3)
	(supports instrument8 infrared1)
	(calibration_target instrument8 groundstation2)
	(supports instrument9 infrared1)
	(supports instrument9 infrared3)
	(calibration_target instrument9 star4)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon5)
	(supports instrument10 infrared1)
	(supports instrument10 spectrograph2)
	(supports instrument10 spectrograph0)
	(calibration_target instrument10 star1)
	(supports instrument11 infrared3)
	(supports instrument11 spectrograph2)
	(calibration_target instrument11 star1)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation2)
)
(:goal (and
	(pointing satellite0 groundstation0)
	(pointing satellite2 groundstation3)
	(have_image phenomenon5 spectrograph0)
	(have_image planet6 spectrograph2)
	(have_image star7 spectrograph0)
	(have_image star8 spectrograph2)
))

)
