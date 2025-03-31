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
	satellite2 - satellite
	instrument6 - instrument
	satellite3 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	satellite4 - satellite
	instrument13 - instrument
	instrument14 - instrument
	image0 - mode
	spectrograph1 - mode
	star1 - direction
	star2 - direction
	star0 - direction
	groundstation3 - direction
	planet4 - direction
	planet5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 star0)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 star1)
	(supports instrument3 spectrograph1)
	(supports instrument3 image0)
	(calibration_target instrument3 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation3)
	(supports instrument4 spectrograph1)
	(supports instrument4 image0)
	(calibration_target instrument4 star0)
	(supports instrument5 spectrograph1)
	(calibration_target instrument5 groundstation3)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument6 image0)
	(supports instrument6 spectrograph1)
	(calibration_target instrument6 star2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon6)
	(supports instrument7 spectrograph1)
	(calibration_target instrument7 star0)
	(supports instrument8 spectrograph1)
	(calibration_target instrument8 star0)
	(supports instrument9 image0)
	(supports instrument9 spectrograph1)
	(calibration_target instrument9 star2)
	(supports instrument10 image0)
	(supports instrument10 spectrograph1)
	(calibration_target instrument10 star0)
	(supports instrument11 image0)
	(calibration_target instrument11 star0)
	(supports instrument12 spectrograph1)
	(supports instrument12 image0)
	(calibration_target instrument12 star2)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation3)
	(supports instrument13 spectrograph1)
	(calibration_target instrument13 star0)
	(supports instrument14 spectrograph1)
	(supports instrument14 image0)
	(calibration_target instrument14 groundstation3)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star1)
)
(:goal (and
	(have_image planet4 image0)
	(have_image planet5 image0)
	(have_image phenomenon6 spectrograph1)
))

)
