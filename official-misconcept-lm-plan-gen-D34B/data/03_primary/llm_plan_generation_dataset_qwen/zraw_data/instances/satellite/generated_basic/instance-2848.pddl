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
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite3 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	image1 - mode
	spectrograph0 - mode
	star2 - direction
	star0 - direction
	groundstation3 - direction
	star4 - direction
	star1 - direction
	star5 - direction
	planet6 - direction
	phenomenon7 - direction
)
(:init
	(supports instrument0 image1)
	(calibration_target instrument0 star1)
	(supports instrument1 image1)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star0)
	(supports instrument2 image1)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 star2)
	(supports instrument3 spectrograph0)
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
	(pointing satellite0 star5)
	(supports instrument5 spectrograph0)
	(supports instrument5 image1)
	(calibration_target instrument5 star0)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 star4)
	(supports instrument7 spectrograph0)
	(supports instrument7 image1)
	(calibration_target instrument7 groundstation3)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 star1)
	(supports instrument9 image1)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 star1)
	(supports instrument10 image1)
	(calibration_target instrument10 groundstation3)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star4)
	(supports instrument11 image1)
	(supports instrument11 spectrograph0)
	(calibration_target instrument11 star4)
	(supports instrument12 spectrograph0)
	(calibration_target instrument12 star4)
	(supports instrument13 image1)
	(calibration_target instrument13 star1)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon7)
)
(:goal (and
	(pointing satellite2 star4)
	(pointing satellite3 phenomenon7)
	(have_image star5 image1)
	(have_image planet6 image1)
	(have_image phenomenon7 image1)
))

)
