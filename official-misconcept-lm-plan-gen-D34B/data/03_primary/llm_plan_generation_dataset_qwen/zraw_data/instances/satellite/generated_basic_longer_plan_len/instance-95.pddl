(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	satellite3 - satellite
	instrument7 - instrument
	instrument8 - instrument
	satellite4 - satellite
	instrument9 - instrument
	satellite5 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	infrared0 - mode
	spectrograph2 - mode
	infrared1 - mode
	star3 - direction
	star0 - direction
	star2 - direction
	groundstation1 - direction
	star4 - direction
	planet5 - direction
	phenomenon6 - direction
	phenomenon7 - direction
	star8 - direction
	phenomenon9 - direction
	star10 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 star2)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument1 infrared1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star0)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star0)
	(supports instrument3 spectrograph2)
	(supports instrument3 infrared1)
	(calibration_target instrument3 star2)
	(supports instrument4 infrared1)
	(supports instrument4 spectrograph2)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star0)
	(supports instrument5 spectrograph2)
	(supports instrument5 infrared1)
	(calibration_target instrument5 star3)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument6 infrared1)
	(supports instrument6 spectrograph2)
	(supports instrument6 infrared0)
	(calibration_target instrument6 groundstation1)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star8)
	(supports instrument7 spectrograph2)
	(supports instrument7 infrared1)
	(calibration_target instrument7 star2)
	(supports instrument8 spectrograph2)
	(supports instrument8 infrared1)
	(supports instrument8 infrared0)
	(calibration_target instrument8 star3)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star4)
	(supports instrument9 infrared1)
	(calibration_target instrument9 star3)
	(on_board instrument9 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet5)
	(supports instrument10 infrared1)
	(calibration_target instrument10 star0)
	(supports instrument11 infrared1)
	(supports instrument11 spectrograph2)
	(supports instrument11 infrared0)
	(calibration_target instrument11 star2)
	(supports instrument12 spectrograph2)
	(supports instrument12 infrared0)
	(calibration_target instrument12 star2)
	(supports instrument13 spectrograph2)
	(supports instrument13 infrared0)
	(supports instrument13 infrared1)
	(calibration_target instrument13 groundstation1)
	(supports instrument14 spectrograph2)
	(calibration_target instrument14 groundstation1)
	(on_board instrument10 satellite5)
	(on_board instrument11 satellite5)
	(on_board instrument12 satellite5)
	(on_board instrument13 satellite5)
	(on_board instrument14 satellite5)
	(power_avail satellite5)
	(pointing satellite5 groundstation1)
)
(:goal (and
	(pointing satellite3 phenomenon6)
	(pointing satellite4 phenomenon7)
	(have_image star4 spectrograph2)
	(have_image planet5 infrared0)
	(have_image phenomenon6 spectrograph2)
	(have_image phenomenon7 infrared0)
	(have_image star8 infrared1)
	(have_image phenomenon9 infrared0)
	(have_image star10 infrared1)
))

)
