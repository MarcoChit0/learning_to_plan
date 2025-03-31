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
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite4 - satellite
	instrument11 - instrument
	satellite5 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	infrared1 - mode
	spectrograph2 - mode
	infrared0 - mode
	groundstation0 - direction
	groundstation3 - direction
	star2 - direction
	groundstation1 - direction
	star4 - direction
	star5 - direction
	phenomenon6 - direction
	phenomenon7 - direction
	star8 - direction
	planet9 - direction
	phenomenon10 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 star2)
	(supports instrument1 spectrograph2)
	(supports instrument1 infrared0)
	(supports instrument1 infrared1)
	(calibration_target instrument1 groundstation3)
	(supports instrument2 infrared0)
	(supports instrument2 infrared1)
	(calibration_target instrument2 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon10)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 spectrograph2)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 spectrograph2)
	(supports instrument5 infrared1)
	(calibration_target instrument5 groundstation0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument6 infrared0)
	(calibration_target instrument6 groundstation3)
	(supports instrument7 infrared1)
	(supports instrument7 infrared0)
	(supports instrument7 spectrograph2)
	(calibration_target instrument7 star2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star5)
	(supports instrument8 infrared0)
	(supports instrument8 infrared1)
	(calibration_target instrument8 groundstation3)
	(supports instrument9 infrared1)
	(supports instrument9 infrared0)
	(calibration_target instrument9 groundstation1)
	(supports instrument10 spectrograph2)
	(calibration_target instrument10 groundstation1)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star8)
	(supports instrument11 spectrograph2)
	(supports instrument11 infrared1)
	(calibration_target instrument11 groundstation3)
	(on_board instrument11 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation1)
	(supports instrument12 spectrograph2)
	(calibration_target instrument12 groundstation0)
	(supports instrument13 infrared1)
	(supports instrument13 infrared0)
	(supports instrument13 spectrograph2)
	(calibration_target instrument13 groundstation3)
	(supports instrument14 infrared0)
	(calibration_target instrument14 star2)
	(supports instrument15 infrared1)
	(calibration_target instrument15 groundstation1)
	(on_board instrument12 satellite5)
	(on_board instrument13 satellite5)
	(on_board instrument14 satellite5)
	(on_board instrument15 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star4)
)
(:goal (and
	(pointing satellite1 groundstation0)
	(have_image star4 spectrograph2)
	(have_image star5 spectrograph2)
	(have_image phenomenon6 infrared0)
	(have_image phenomenon7 infrared1)
	(have_image star8 infrared0)
	(have_image planet9 spectrograph2)
	(have_image phenomenon10 infrared0)
))

)
