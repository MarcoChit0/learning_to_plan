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
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	instrument10 - instrument
	image1 - mode
	image2 - mode
	spectrograph0 - mode
	star1 - direction
	star0 - direction
	planet2 - direction
	planet3 - direction
)
(:init
	(supports instrument0 image2)
	(calibration_target instrument0 star1)
	(supports instrument1 image2)
	(supports instrument1 image1)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star1)
	(supports instrument2 image2)
	(supports instrument2 spectrograph0)
	(supports instrument2 image1)
	(calibration_target instrument2 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument3 image1)
	(calibration_target instrument3 star0)
	(supports instrument4 image1)
	(supports instrument4 image2)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 star1)
	(supports instrument5 image1)
	(supports instrument5 image2)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 star1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument6 spectrograph0)
	(supports instrument6 image1)
	(supports instrument6 image2)
	(calibration_target instrument6 star1)
	(supports instrument7 spectrograph0)
	(supports instrument7 image1)
	(calibration_target instrument7 star1)
	(supports instrument8 image1)
	(supports instrument8 image2)
	(calibration_target instrument8 star0)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet3)
	(supports instrument9 spectrograph0)
	(supports instrument9 image2)
	(supports instrument9 image1)
	(calibration_target instrument9 star0)
	(supports instrument10 image1)
	(supports instrument10 image2)
	(calibration_target instrument10 star0)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet3)
)
(:goal (and
	(pointing satellite0 planet3)
	(pointing satellite1 planet2)
	(pointing satellite3 star1)
	(have_image planet2 image2)
	(have_image planet3 spectrograph0)
))

)
