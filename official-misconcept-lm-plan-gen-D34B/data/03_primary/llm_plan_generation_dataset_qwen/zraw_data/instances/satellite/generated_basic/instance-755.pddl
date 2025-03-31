(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	satellite2 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	satellite3 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	spectrograph2 - mode
	image0 - mode
	image1 - mode
	star0 - direction
	star3 - direction
	star1 - direction
	star5 - direction
	groundstation2 - direction
	star4 - direction
	planet6 - direction
	phenomenon7 - direction
	planet8 - direction
	star9 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 star4)
	(calibration_target instrument0 star1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument1 image1)
	(supports instrument1 spectrograph2)
	(supports instrument1 image0)
	(calibration_target instrument1 star0)
	(calibration_target instrument1 star4)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon7)
	(supports instrument2 image1)
	(supports instrument2 image0)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 star5)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 spectrograph2)
	(supports instrument3 image0)
	(calibration_target instrument3 star5)
	(supports instrument4 image1)
	(calibration_target instrument4 star1)
	(supports instrument5 image0)
	(supports instrument5 image1)
	(calibration_target instrument5 star3)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 image1)
	(supports instrument6 image0)
	(supports instrument6 spectrograph2)
	(calibration_target instrument6 groundstation2)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet8)
	(supports instrument7 spectrograph2)
	(supports instrument7 image0)
	(calibration_target instrument7 star5)
	(calibration_target instrument7 star1)
	(supports instrument8 image0)
	(calibration_target instrument8 star4)
	(calibration_target instrument8 groundstation2)
	(supports instrument9 image0)
	(supports instrument9 spectrograph2)
	(supports instrument9 image1)
	(calibration_target instrument9 star4)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet6)
)
(:goal (and
	(pointing satellite2 star1)
	(have_image planet6 spectrograph2)
	(have_image phenomenon7 image0)
	(have_image planet8 image0)
	(have_image star9 image0)
))

)
