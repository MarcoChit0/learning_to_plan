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
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	image0 - mode
	spectrograph1 - mode
	image2 - mode
	groundstation0 - direction
	groundstation2 - direction
	star1 - direction
	planet3 - direction
	planet4 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(supports instrument0 image2)
	(supports instrument0 image0)
	(calibration_target instrument0 star1)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 star1)
	(supports instrument2 spectrograph1)
	(supports instrument2 image2)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation2)
	(supports instrument5 spectrograph1)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 image0)
	(supports instrument6 spectrograph1)
	(calibration_target instrument6 groundstation0)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument7 image0)
	(supports instrument7 image2)
	(supports instrument7 spectrograph1)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 spectrograph1)
	(supports instrument8 image0)
	(supports instrument8 image2)
	(calibration_target instrument8 star1)
	(supports instrument9 image2)
	(supports instrument9 image0)
	(calibration_target instrument9 groundstation0)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
	(supports instrument10 image2)
	(calibration_target instrument10 star1)
	(supports instrument11 image2)
	(calibration_target instrument11 star1)
	(supports instrument12 image2)
	(supports instrument12 spectrograph1)
	(supports instrument12 image0)
	(calibration_target instrument12 groundstation2)
	(supports instrument13 image2)
	(supports instrument13 spectrograph1)
	(calibration_target instrument13 groundstation2)
	(supports instrument14 image2)
	(supports instrument14 spectrograph1)
	(supports instrument14 image0)
	(calibration_target instrument14 star1)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet4)
)
(:goal (and
	(have_image planet3 image0)
	(have_image planet4 image0)
))

)
