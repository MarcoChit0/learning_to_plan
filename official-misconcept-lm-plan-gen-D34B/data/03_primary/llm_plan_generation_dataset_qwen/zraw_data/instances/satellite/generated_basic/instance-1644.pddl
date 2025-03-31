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
	instrument9 - instrument
	image1 - mode
	thermograph0 - mode
	spectrograph2 - mode
	groundstation0 - direction
	star1 - direction
	star2 - direction
	star3 - direction
	phenomenon4 - direction
)
(:init
	(supports instrument0 thermograph0)
	(supports instrument0 spectrograph2)
	(supports instrument0 image1)
	(calibration_target instrument0 star1)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument3 thermograph0)
	(supports instrument3 image1)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 image1)
	(supports instrument5 spectrograph2)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 groundstation0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon4)
	(supports instrument6 thermograph0)
	(supports instrument6 image1)
	(calibration_target instrument6 star1)
	(supports instrument7 thermograph0)
	(supports instrument7 image1)
	(calibration_target instrument7 star1)
	(supports instrument8 thermograph0)
	(supports instrument8 image1)
	(calibration_target instrument8 star1)
	(supports instrument9 thermograph0)
	(supports instrument9 image1)
	(calibration_target instrument9 star1)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
)
(:goal (and
	(have_image star2 image1)
	(have_image star3 thermograph0)
	(have_image phenomenon4 thermograph0)
))

)
