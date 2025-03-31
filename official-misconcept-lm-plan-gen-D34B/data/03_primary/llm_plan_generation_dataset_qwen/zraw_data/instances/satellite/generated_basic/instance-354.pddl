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
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	image0 - mode
	thermograph1 - mode
	groundstation0 - direction
	star2 - direction
	groundstation1 - direction
	star3 - direction
	planet4 - direction
	phenomenon5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 thermograph1)
	(calibration_target instrument0 star2)
	(supports instrument1 image0)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 star2)
	(supports instrument2 image0)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon5)
	(supports instrument3 thermograph1)
	(supports instrument3 image0)
	(calibration_target instrument3 star2)
	(supports instrument4 image0)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 star2)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet4)
	(supports instrument5 image0)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 image0)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 star3)
	(supports instrument7 image0)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 groundstation1)
	(supports instrument8 image0)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 star3)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon6)
)
(:goal (and
	(pointing satellite1 groundstation1)
	(have_image planet4 thermograph1)
	(have_image phenomenon5 image0)
	(have_image phenomenon6 image0)
))

)
