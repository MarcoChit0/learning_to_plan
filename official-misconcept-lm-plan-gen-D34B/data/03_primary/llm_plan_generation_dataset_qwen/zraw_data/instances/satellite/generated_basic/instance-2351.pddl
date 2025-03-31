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
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite2 - satellite
	instrument9 - instrument
	thermograph1 - mode
	image0 - mode
	groundstation0 - direction
	star1 - direction
	phenomenon2 - direction
	phenomenon3 - direction
	star4 - direction
	star5 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 thermograph1)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 image0)
	(calibration_target instrument1 star1)
	(supports instrument2 thermograph1)
	(supports instrument2 image0)
	(calibration_target instrument2 star1)
	(supports instrument3 image0)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument4 thermograph1)
	(supports instrument4 image0)
	(calibration_target instrument4 star1)
	(supports instrument5 image0)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 image0)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 star1)
	(supports instrument7 image0)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 thermograph1)
	(supports instrument8 image0)
	(calibration_target instrument8 star1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star5)
	(supports instrument9 thermograph1)
	(supports instrument9 image0)
	(calibration_target instrument9 star1)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star4)
)
(:goal (and
	(have_image phenomenon2 image0)
	(have_image phenomenon3 image0)
	(have_image star4 thermograph1)
	(have_image star5 image0)
))

)
