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
	satellite3 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	spectrograph2 - mode
	image3 - mode
	image1 - mode
	thermograph0 - mode
	groundstation2 - direction
	star4 - direction
	star0 - direction
	star1 - direction
	star3 - direction
	phenomenon5 - direction
	planet6 - direction
	phenomenon7 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star4)
	(supports instrument1 image1)
	(calibration_target instrument1 star0)
	(supports instrument2 thermograph0)
	(supports instrument2 image1)
	(calibration_target instrument2 star1)
	(supports instrument3 image1)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star4)
	(supports instrument4 spectrograph2)
	(supports instrument4 image3)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 star1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument6 thermograph0)
	(supports instrument6 spectrograph2)
	(calibration_target instrument6 star0)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
	(supports instrument7 spectrograph2)
	(supports instrument7 image3)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star3)
	(supports instrument8 spectrograph2)
	(calibration_target instrument8 star1)
	(supports instrument9 spectrograph2)
	(supports instrument9 image1)
	(calibration_target instrument9 star3)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star3)
)
(:goal (and
	(pointing satellite1 groundstation2)
	(pointing satellite2 star4)
	(have_image phenomenon5 image3)
	(have_image planet6 image3)
	(have_image phenomenon7 thermograph0)
))

)
