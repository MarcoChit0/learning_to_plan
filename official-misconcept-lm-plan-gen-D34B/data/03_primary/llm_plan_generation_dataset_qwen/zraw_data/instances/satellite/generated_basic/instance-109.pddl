(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	satellite3 - satellite
	instrument4 - instrument
	instrument5 - instrument
	satellite4 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	thermograph0 - mode
	image1 - mode
	star3 - direction
	star0 - direction
	star1 - direction
	groundstation5 - direction
	groundstation4 - direction
	groundstation2 - direction
	phenomenon6 - direction
	star7 - direction
	phenomenon8 - direction
	phenomenon9 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon8)
	(supports instrument1 image1)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 star0)
	(supports instrument2 thermograph0)
	(supports instrument2 image1)
	(calibration_target instrument2 star3)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon6)
	(supports instrument3 image1)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star3)
	(calibration_target instrument3 star0)
	(on_board instrument3 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star7)
	(supports instrument4 thermograph0)
	(supports instrument4 image1)
	(calibration_target instrument4 groundstation5)
	(supports instrument5 thermograph0)
	(supports instrument5 image1)
	(calibration_target instrument5 groundstation5)
	(on_board instrument4 satellite3)
	(on_board instrument5 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation5)
	(supports instrument6 image1)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 star0)
	(calibration_target instrument6 groundstation5)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star1)
	(supports instrument8 image1)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 star1)
	(supports instrument9 image1)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 groundstation4)
	(calibration_target instrument9 groundstation5)
	(supports instrument10 thermograph0)
	(calibration_target instrument10 groundstation2)
	(supports instrument11 thermograph0)
	(supports instrument11 image1)
	(calibration_target instrument11 groundstation2)
	(on_board instrument6 satellite4)
	(on_board instrument7 satellite4)
	(on_board instrument8 satellite4)
	(on_board instrument9 satellite4)
	(on_board instrument10 satellite4)
	(on_board instrument11 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation2)
)
(:goal (and
	(pointing satellite1 star0)
	(pointing satellite2 star7)
	(pointing satellite3 star7)
	(have_image phenomenon6 thermograph0)
	(have_image star7 thermograph0)
	(have_image phenomenon8 thermograph0)
	(have_image phenomenon9 thermograph0)
))

)
