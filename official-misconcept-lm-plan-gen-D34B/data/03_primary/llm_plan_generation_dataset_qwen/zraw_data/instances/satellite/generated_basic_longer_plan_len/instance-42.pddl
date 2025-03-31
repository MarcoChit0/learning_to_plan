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
	satellite3 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	thermograph1 - mode
	thermograph2 - mode
	thermograph3 - mode
	thermograph0 - mode
	groundstation0 - direction
	star2 - direction
	star1 - direction
	star3 - direction
	phenomenon4 - direction
	phenomenon5 - direction
	star6 - direction
	phenomenon7 - direction
	star8 - direction
	star9 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 thermograph3)
	(supports instrument0 thermograph2)
	(calibration_target instrument0 star3)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 star3)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 star2)
	(supports instrument3 thermograph0)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star9)
	(supports instrument4 thermograph3)
	(calibration_target instrument4 star3)
	(supports instrument5 thermograph2)
	(supports instrument5 thermograph3)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 star2)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon4)
	(supports instrument6 thermograph2)
	(calibration_target instrument6 star1)
	(supports instrument7 thermograph0)
	(supports instrument7 thermograph3)
	(supports instrument7 thermograph2)
	(calibration_target instrument7 star2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
	(supports instrument8 thermograph3)
	(supports instrument8 thermograph2)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 star1)
	(supports instrument9 thermograph0)
	(supports instrument9 thermograph2)
	(calibration_target instrument9 star3)
	(supports instrument10 thermograph2)
	(supports instrument10 thermograph1)
	(calibration_target instrument10 star1)
	(supports instrument11 thermograph1)
	(supports instrument11 thermograph0)
	(supports instrument11 thermograph2)
	(calibration_target instrument11 star3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star1)
)
(:goal (and
	(pointing satellite0 star1)
	(have_image phenomenon4 thermograph0)
	(have_image phenomenon5 thermograph3)
	(have_image star6 thermograph2)
	(have_image phenomenon7 thermograph1)
	(have_image star8 thermograph2)
	(have_image star9 thermograph0)
))

)
