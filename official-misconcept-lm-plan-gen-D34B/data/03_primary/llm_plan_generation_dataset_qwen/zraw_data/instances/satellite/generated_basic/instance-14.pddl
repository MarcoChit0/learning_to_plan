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
	instrument11 - instrument
	instrument12 - instrument
	thermograph0 - mode
	image1 - mode
	groundstation0 - direction
	star1 - direction
	phenomenon2 - direction
	planet3 - direction
	planet4 - direction
	star5 - direction
)
(:init
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 thermograph0)
	(supports instrument1 image1)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 thermograph0)
	(supports instrument2 image1)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 image1)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 thermograph0)
	(supports instrument4 image1)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 image1)
	(calibration_target instrument5 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star5)
	(supports instrument6 thermograph0)
	(supports instrument6 image1)
	(calibration_target instrument6 groundstation0)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star5)
	(supports instrument7 thermograph0)
	(supports instrument7 image1)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 image1)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 image1)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 thermograph0)
	(supports instrument10 image1)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 thermograph0)
	(supports instrument11 image1)
	(calibration_target instrument11 groundstation0)
	(supports instrument12 thermograph0)
	(calibration_target instrument12 groundstation0)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet3)
)
(:goal (and
	(pointing satellite0 planet4)
	(pointing satellite1 planet4)
	(have_image star1 image1)
	(have_image phenomenon2 thermograph0)
	(have_image planet3 thermograph0)
	(have_image planet4 thermograph0)
	(have_image star5 thermograph0)
))

)
