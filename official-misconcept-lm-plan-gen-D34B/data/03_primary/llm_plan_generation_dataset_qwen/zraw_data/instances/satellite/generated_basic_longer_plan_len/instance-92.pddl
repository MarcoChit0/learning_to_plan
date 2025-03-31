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
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	thermograph1 - mode
	infrared2 - mode
	thermograph3 - mode
	image0 - mode
	groundstation0 - direction
	groundstation3 - direction
	groundstation2 - direction
	groundstation1 - direction
	phenomenon4 - direction
	star5 - direction
	star6 - direction
	star7 - direction
	phenomenon8 - direction
	phenomenon9 - direction
)
(:init
	(supports instrument0 thermograph1)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 infrared2)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star5)
	(supports instrument2 thermograph3)
	(supports instrument2 image0)
	(supports instrument2 infrared2)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 thermograph3)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 groundstation3)
	(supports instrument5 infrared2)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 image0)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 image0)
	(calibration_target instrument7 groundstation3)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon8)
	(supports instrument8 thermograph1)
	(supports instrument8 image0)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 thermograph1)
	(supports instrument9 infrared2)
	(supports instrument9 thermograph3)
	(calibration_target instrument9 groundstation3)
	(supports instrument10 thermograph3)
	(calibration_target instrument10 groundstation2)
	(supports instrument11 image0)
	(calibration_target instrument11 groundstation2)
	(supports instrument12 thermograph3)
	(supports instrument12 image0)
	(calibration_target instrument12 groundstation1)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star5)
)
(:goal (and
	(pointing satellite1 groundstation3)
	(pointing satellite2 star7)
	(have_image phenomenon4 image0)
	(have_image star5 image0)
	(have_image star6 thermograph3)
	(have_image star7 infrared2)
	(have_image phenomenon8 thermograph1)
	(have_image phenomenon9 thermograph1)
))

)
