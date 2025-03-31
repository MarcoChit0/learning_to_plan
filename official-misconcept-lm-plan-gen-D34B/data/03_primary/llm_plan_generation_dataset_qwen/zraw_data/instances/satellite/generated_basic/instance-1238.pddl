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
	satellite3 - satellite
	instrument9 - instrument
	image1 - mode
	thermograph3 - mode
	image0 - mode
	spectrograph2 - mode
	star2 - direction
	star3 - direction
	groundstation1 - direction
	groundstation0 - direction
	groundstation4 - direction
	star5 - direction
	planet6 - direction
	star7 - direction
	star8 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 spectrograph2)
	(supports instrument0 thermograph3)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 thermograph3)
	(supports instrument1 spectrograph2)
	(supports instrument1 image1)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 spectrograph2)
	(supports instrument2 image0)
	(supports instrument2 thermograph3)
	(calibration_target instrument2 star3)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 groundstation4)
	(supports instrument4 thermograph3)
	(supports instrument4 image1)
	(calibration_target instrument4 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation4)
	(supports instrument5 image1)
	(calibration_target instrument5 groundstation4)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument6 image1)
	(supports instrument6 image0)
	(supports instrument6 thermograph3)
	(calibration_target instrument6 groundstation4)
	(supports instrument7 spectrograph2)
	(supports instrument7 image1)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 image0)
	(calibration_target instrument8 groundstation4)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
	(supports instrument9 thermograph3)
	(supports instrument9 image1)
	(supports instrument9 spectrograph2)
	(calibration_target instrument9 groundstation4)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation0)
)
(:goal (and
	(pointing satellite2 groundstation0)
	(have_image star5 image0)
	(have_image planet6 image0)
	(have_image star7 image0)
	(have_image star8 image1)
))

)
