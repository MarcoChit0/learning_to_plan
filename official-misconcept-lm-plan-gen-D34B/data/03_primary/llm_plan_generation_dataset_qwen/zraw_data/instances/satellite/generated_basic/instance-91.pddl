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
	instrument6 - instrument
	instrument7 - instrument
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	satellite3 - satellite
	instrument13 - instrument
	satellite4 - satellite
	instrument14 - instrument
	spectrograph1 - mode
	image0 - mode
	star5 - direction
	star2 - direction
	groundstation1 - direction
	star4 - direction
	star0 - direction
	star3 - direction
	star6 - direction
	phenomenon7 - direction
	star8 - direction
	phenomenon9 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star0)
	(supports instrument1 image0)
	(calibration_target instrument1 star2)
	(calibration_target instrument1 star4)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 groundstation1)
	(calibration_target instrument2 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon9)
	(supports instrument3 image0)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 star4)
	(supports instrument4 image0)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 star4)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 image0)
	(supports instrument5 spectrograph1)
	(calibration_target instrument5 star0)
	(calibration_target instrument5 star5)
	(supports instrument6 spectrograph1)
	(supports instrument6 image0)
	(calibration_target instrument6 star2)
	(supports instrument7 image0)
	(supports instrument7 spectrograph1)
	(calibration_target instrument7 groundstation1)
	(calibration_target instrument7 star5)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
	(supports instrument8 spectrograph1)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 image0)
	(supports instrument9 spectrograph1)
	(calibration_target instrument9 star2)
	(supports instrument10 spectrograph1)
	(supports instrument10 image0)
	(calibration_target instrument10 star2)
	(calibration_target instrument10 star0)
	(supports instrument11 spectrograph1)
	(calibration_target instrument11 star4)
	(calibration_target instrument11 groundstation1)
	(supports instrument12 image0)
	(supports instrument12 spectrograph1)
	(calibration_target instrument12 star4)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star8)
	(supports instrument13 spectrograph1)
	(calibration_target instrument13 star0)
	(on_board instrument13 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon9)
	(supports instrument14 spectrograph1)
	(supports instrument14 image0)
	(calibration_target instrument14 star3)
	(on_board instrument14 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star3)
)
(:goal (and
	(pointing satellite1 star3)
	(have_image star6 spectrograph1)
	(have_image phenomenon7 image0)
	(have_image star8 spectrograph1)
	(have_image phenomenon9 spectrograph1)
))

)
