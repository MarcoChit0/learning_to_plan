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
	satellite3 - satellite
	instrument5 - instrument
	instrument6 - instrument
	satellite4 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite5 - satellite
	instrument10 - instrument
	instrument11 - instrument
	thermograph0 - mode
	image1 - mode
	thermograph2 - mode
	groundstation2 - direction
	star0 - direction
	star1 - direction
	star3 - direction
	planet4 - direction
	planet5 - direction
	star6 - direction
	phenomenon7 - direction
	planet8 - direction
	phenomenon9 - direction
	star10 - direction
)
(:init
	(supports instrument0 thermograph2)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet4)
	(supports instrument1 thermograph0)
	(supports instrument1 thermograph2)
	(supports instrument1 image1)
	(calibration_target instrument1 star3)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet5)
	(supports instrument2 thermograph0)
	(supports instrument2 image1)
	(calibration_target instrument2 star1)
	(supports instrument3 image1)
	(supports instrument3 thermograph2)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star0)
	(supports instrument4 thermograph2)
	(calibration_target instrument4 star3)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
	(supports instrument5 image1)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 image1)
	(calibration_target instrument6 star0)
	(on_board instrument5 satellite3)
	(on_board instrument6 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star3)
	(supports instrument7 image1)
	(calibration_target instrument7 star0)
	(supports instrument8 thermograph0)
	(supports instrument8 image1)
	(supports instrument8 thermograph2)
	(calibration_target instrument8 star0)
	(supports instrument9 thermograph2)
	(supports instrument9 image1)
	(calibration_target instrument9 star0)
	(on_board instrument7 satellite4)
	(on_board instrument8 satellite4)
	(on_board instrument9 satellite4)
	(power_avail satellite4)
	(pointing satellite4 phenomenon7)
	(supports instrument10 thermograph2)
	(supports instrument10 thermograph0)
	(calibration_target instrument10 star1)
	(supports instrument11 thermograph2)
	(supports instrument11 image1)
	(calibration_target instrument11 star3)
	(on_board instrument10 satellite5)
	(on_board instrument11 satellite5)
	(power_avail satellite5)
	(pointing satellite5 phenomenon9)
)
(:goal (and
	(pointing satellite1 star0)
	(pointing satellite2 groundstation2)
	(pointing satellite5 star0)
	(have_image planet4 thermograph2)
	(have_image planet5 image1)
	(have_image star6 image1)
	(have_image phenomenon7 thermograph2)
	(have_image planet8 thermograph0)
	(have_image phenomenon9 image1)
	(have_image star10 thermograph2)
))

)
