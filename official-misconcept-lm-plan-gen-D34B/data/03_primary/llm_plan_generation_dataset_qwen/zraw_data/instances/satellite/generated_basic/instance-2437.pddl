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
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	infrared1 - mode
	image0 - mode
	thermograph2 - mode
	groundstation0 - direction
	groundstation1 - direction
	planet2 - direction
	phenomenon3 - direction
	phenomenon4 - direction
	planet5 - direction
)
(:init
	(supports instrument0 thermograph2)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 infrared1)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 image0)
	(supports instrument2 thermograph2)
	(supports instrument2 infrared1)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 thermograph2)
	(supports instrument3 infrared1)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 thermograph2)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument6 infrared1)
	(supports instrument6 thermograph2)
	(supports instrument6 image0)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 image0)
	(calibration_target instrument7 groundstation1)
	(supports instrument8 image0)
	(supports instrument8 thermograph2)
	(supports instrument8 infrared1)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 image0)
	(supports instrument9 thermograph2)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 infrared1)
	(supports instrument10 image0)
	(calibration_target instrument10 groundstation1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(on_board instrument10 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon3)
)
(:goal (and
	(pointing satellite0 planet2)
	(pointing satellite1 planet2)
	(have_image planet2 image0)
	(have_image phenomenon3 thermograph2)
	(have_image phenomenon4 infrared1)
	(have_image planet5 infrared1)
))

)
