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
	instrument7 - instrument
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite3 - satellite
	instrument11 - instrument
	spectrograph3 - mode
	image2 - mode
	spectrograph1 - mode
	spectrograph0 - mode
	groundstation0 - direction
	star1 - direction
	planet2 - direction
	planet3 - direction
)
(:init
	(supports instrument0 spectrograph3)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 spectrograph0)
	(supports instrument2 image2)
	(supports instrument2 spectrograph3)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 spectrograph1)
	(supports instrument3 spectrograph3)
	(calibration_target instrument3 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet3)
	(supports instrument4 spectrograph3)
	(supports instrument4 spectrograph1)
	(supports instrument4 image2)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 image2)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 image2)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 spectrograph3)
	(supports instrument7 spectrograph1)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 groundstation0)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet3)
	(supports instrument8 spectrograph0)
	(supports instrument8 spectrograph3)
	(supports instrument8 image2)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 spectrograph0)
	(supports instrument9 spectrograph1)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 spectrograph1)
	(supports instrument10 spectrograph0)
	(supports instrument10 spectrograph3)
	(calibration_target instrument10 groundstation0)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet3)
	(supports instrument11 spectrograph0)
	(supports instrument11 spectrograph3)
	(supports instrument11 image2)
	(calibration_target instrument11 groundstation0)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet2)
)
(:goal (and
	(pointing satellite0 planet2)
	(pointing satellite1 star1)
	(pointing satellite2 planet3)
	(have_image star1 spectrograph1)
	(have_image planet2 image2)
	(have_image planet3 image2)
))

)
