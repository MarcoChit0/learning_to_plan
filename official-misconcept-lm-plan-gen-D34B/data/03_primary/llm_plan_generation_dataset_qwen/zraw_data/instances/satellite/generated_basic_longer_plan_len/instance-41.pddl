(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	image1 - mode
	thermograph0 - mode
	image3 - mode
	thermograph2 - mode
	star3 - direction
	star2 - direction
	star0 - direction
	groundstation1 - direction
	star4 - direction
	planet5 - direction
	planet6 - direction
	star7 - direction
	star8 - direction
	phenomenon9 - direction
)
(:init
	(supports instrument0 thermograph0)
	(supports instrument0 image1)
	(supports instrument0 thermograph2)
	(calibration_target instrument0 star0)
	(supports instrument1 image1)
	(calibration_target instrument1 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument2 image3)
	(calibration_target instrument2 star3)
	(supports instrument3 image1)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 image1)
	(supports instrument4 thermograph2)
	(supports instrument4 image3)
	(calibration_target instrument4 star0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star8)
	(supports instrument5 image1)
	(supports instrument5 thermograph2)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 thermograph2)
	(calibration_target instrument6 star3)
	(supports instrument7 image3)
	(calibration_target instrument7 star2)
	(supports instrument8 image3)
	(supports instrument8 image1)
	(calibration_target instrument8 star2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
	(supports instrument9 thermograph2)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 star0)
	(supports instrument10 thermograph2)
	(calibration_target instrument10 star2)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 star0)
	(supports instrument12 image3)
	(calibration_target instrument12 groundstation1)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet6)
)
(:goal (and
	(pointing satellite2 phenomenon9)
	(pointing satellite3 star7)
	(have_image star4 thermograph2)
	(have_image planet5 thermograph2)
	(have_image planet6 thermograph2)
	(have_image star7 image1)
	(have_image star8 image1)
	(have_image phenomenon9 image3)
))

)
