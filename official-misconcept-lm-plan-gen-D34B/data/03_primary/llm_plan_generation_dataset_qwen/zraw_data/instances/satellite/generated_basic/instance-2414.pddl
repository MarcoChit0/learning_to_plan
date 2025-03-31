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
	instrument7 - instrument
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	spectrograph2 - mode
	image1 - mode
	image0 - mode
	image3 - mode
	star4 - direction
	star2 - direction
	groundstation3 - direction
	star1 - direction
	groundstation0 - direction
	planet5 - direction
	planet6 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 spectrograph2)
	(supports instrument0 image3)
	(calibration_target instrument0 star2)
	(supports instrument1 image1)
	(calibration_target instrument1 star1)
	(supports instrument2 spectrograph2)
	(supports instrument2 image3)
	(calibration_target instrument2 star2)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet6)
	(supports instrument4 image1)
	(supports instrument4 image3)
	(calibration_target instrument4 star2)
	(supports instrument5 image1)
	(supports instrument5 spectrograph2)
	(supports instrument5 image3)
	(calibration_target instrument5 groundstation0)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument6 image1)
	(supports instrument6 spectrograph2)
	(supports instrument6 image3)
	(calibration_target instrument6 star2)
	(supports instrument7 spectrograph2)
	(supports instrument7 image0)
	(supports instrument7 image3)
	(calibration_target instrument7 star1)
	(supports instrument8 image1)
	(supports instrument8 spectrograph2)
	(calibration_target instrument8 groundstation3)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star4)
	(supports instrument9 image0)
	(supports instrument9 image3)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 spectrograph2)
	(supports instrument10 image3)
	(supports instrument10 image0)
	(calibration_target instrument10 star1)
	(supports instrument11 image3)
	(supports instrument11 image0)
	(supports instrument11 image1)
	(calibration_target instrument11 groundstation0)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet5)
)
(:goal (and
	(have_image planet5 image0)
	(have_image planet6 spectrograph2)
))

)
