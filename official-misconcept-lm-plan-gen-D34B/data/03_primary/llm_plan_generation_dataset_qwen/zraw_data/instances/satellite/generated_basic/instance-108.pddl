(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite1 - satellite
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite3 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	satellite4 - satellite
	instrument15 - instrument
	thermograph0 - mode
	image1 - mode
	star0 - direction
	star1 - direction
	star5 - direction
	groundstation3 - direction
	groundstation4 - direction
	groundstation2 - direction
	star6 - direction
	phenomenon7 - direction
	star8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 thermograph0)
	(supports instrument0 image1)
	(calibration_target instrument0 star5)
	(calibration_target instrument0 star1)
	(supports instrument1 thermograph0)
	(supports instrument1 image1)
	(calibration_target instrument1 star0)
	(calibration_target instrument1 groundstation4)
	(supports instrument2 image1)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 groundstation2)
	(calibration_target instrument2 groundstation3)
	(supports instrument3 image1)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 groundstation4)
	(calibration_target instrument3 star0)
	(supports instrument4 image1)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 star5)
	(calibration_target instrument4 groundstation2)
	(supports instrument5 thermograph0)
	(supports instrument5 image1)
	(calibration_target instrument5 groundstation4)
	(calibration_target instrument5 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star8)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 groundstation4)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument7 image1)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star1)
	(supports instrument8 thermograph0)
	(supports instrument8 image1)
	(calibration_target instrument8 star5)
	(calibration_target instrument8 star1)
	(supports instrument9 image1)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 star0)
	(calibration_target instrument9 star1)
	(supports instrument10 image1)
	(supports instrument10 thermograph0)
	(calibration_target instrument10 star0)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
	(supports instrument11 image1)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 groundstation3)
	(supports instrument12 thermograph0)
	(supports instrument12 image1)
	(calibration_target instrument12 star5)
	(calibration_target instrument12 star1)
	(supports instrument13 thermograph0)
	(supports instrument13 image1)
	(calibration_target instrument13 groundstation3)
	(calibration_target instrument13 groundstation4)
	(supports instrument14 thermograph0)
	(calibration_target instrument14 groundstation4)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star5)
	(supports instrument15 thermograph0)
	(supports instrument15 image1)
	(calibration_target instrument15 groundstation2)
	(on_board instrument15 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation3)
)
(:goal (and
	(pointing satellite3 groundstation2)
	(have_image star6 thermograph0)
	(have_image phenomenon7 image1)
	(have_image star8 image1)
	(have_image planet9 image1)
))

)
