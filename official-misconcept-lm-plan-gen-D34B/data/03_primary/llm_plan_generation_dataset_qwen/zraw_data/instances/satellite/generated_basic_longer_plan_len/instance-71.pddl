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
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	satellite4 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	spectrograph2 - mode
	image0 - mode
	image3 - mode
	infrared1 - mode
	star0 - direction
	groundstation2 - direction
	groundstation1 - direction
	planet3 - direction
	planet4 - direction
	planet5 - direction
	star6 - direction
	planet7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star0)
	(supports instrument1 spectrograph2)
	(supports instrument1 image3)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 infrared1)
	(calibration_target instrument2 star0)
	(supports instrument3 image0)
	(supports instrument3 infrared1)
	(calibration_target instrument3 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet5)
	(supports instrument4 image0)
	(calibration_target instrument4 star0)
	(supports instrument5 image0)
	(supports instrument5 infrared1)
	(supports instrument5 spectrograph2)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 spectrograph2)
	(calibration_target instrument6 star0)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star6)
	(supports instrument7 image3)
	(supports instrument7 infrared1)
	(calibration_target instrument7 groundstation1)
	(supports instrument8 infrared1)
	(calibration_target instrument8 groundstation1)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet4)
	(supports instrument9 image0)
	(supports instrument9 infrared1)
	(supports instrument9 spectrograph2)
	(calibration_target instrument9 star0)
	(supports instrument10 image3)
	(supports instrument10 spectrograph2)
	(supports instrument10 image0)
	(calibration_target instrument10 groundstation2)
	(supports instrument11 spectrograph2)
	(supports instrument11 image0)
	(supports instrument11 infrared1)
	(calibration_target instrument11 groundstation2)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet5)
	(supports instrument12 image3)
	(calibration_target instrument12 groundstation1)
	(supports instrument13 image0)
	(supports instrument13 infrared1)
	(calibration_target instrument13 groundstation1)
	(supports instrument14 image0)
	(calibration_target instrument14 groundstation1)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet5)
)
(:goal (and
	(pointing satellite0 planet4)
	(pointing satellite1 groundstation1)
	(pointing satellite3 star0)
	(have_image planet3 image0)
	(have_image planet4 infrared1)
	(have_image planet5 image3)
	(have_image star6 image3)
	(have_image planet7 infrared1)
	(have_image phenomenon8 infrared1)
))

)
