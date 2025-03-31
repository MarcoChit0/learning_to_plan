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
	satellite2 - satellite
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	instrument9 - instrument
	satellite4 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	image1 - mode
	spectrograph0 - mode
	star4 - direction
	star3 - direction
	groundstation0 - direction
	star1 - direction
	star2 - direction
	star5 - direction
	phenomenon6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 image1)
	(calibration_target instrument0 star1)
	(supports instrument1 image1)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star3)
	(supports instrument2 image1)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 spectrograph0)
	(supports instrument3 image1)
	(calibration_target instrument3 star4)
	(supports instrument4 spectrograph0)
	(supports instrument4 image1)
	(calibration_target instrument4 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon6)
	(supports instrument5 image1)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 star2)
	(supports instrument6 spectrograph0)
	(supports instrument6 image1)
	(calibration_target instrument6 star1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument7 spectrograph0)
	(supports instrument7 image1)
	(calibration_target instrument7 groundstation0)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star4)
	(supports instrument8 image1)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 star3)
	(supports instrument9 image1)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 groundstation0)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon7)
	(supports instrument10 image1)
	(calibration_target instrument10 star2)
	(supports instrument11 image1)
	(calibration_target instrument11 groundstation0)
	(supports instrument12 spectrograph0)
	(calibration_target instrument12 star2)
	(supports instrument13 spectrograph0)
	(calibration_target instrument13 star1)
	(supports instrument14 image1)
	(supports instrument14 spectrograph0)
	(calibration_target instrument14 star2)
	(on_board instrument10 satellite4)
	(on_board instrument11 satellite4)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star2)
)
(:goal (and
	(pointing satellite0 groundstation0)
	(have_image star5 spectrograph0)
	(have_image phenomenon6 spectrograph0)
	(have_image phenomenon7 spectrograph0)
	(have_image phenomenon8 spectrograph0)
))

)
