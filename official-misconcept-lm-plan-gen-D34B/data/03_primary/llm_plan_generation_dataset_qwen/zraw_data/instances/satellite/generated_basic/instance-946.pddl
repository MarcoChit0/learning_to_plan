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
	satellite2 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	thermograph1 - mode
	image3 - mode
	infrared2 - mode
	image0 - mode
	groundstation3 - direction
	groundstation1 - direction
	groundstation0 - direction
	groundstation2 - direction
	planet4 - direction
	planet5 - direction
	planet6 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 infrared2)
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 thermograph1)
	(supports instrument1 infrared2)
	(supports instrument1 image3)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 infrared2)
	(supports instrument2 image3)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 thermograph1)
	(supports instrument4 image0)
	(supports instrument4 image3)
	(calibration_target instrument4 groundstation3)
	(supports instrument5 thermograph1)
	(supports instrument5 image0)
	(supports instrument5 infrared2)
	(calibration_target instrument5 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet4)
	(supports instrument6 infrared2)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 groundstation2)
	(supports instrument7 thermograph1)
	(supports instrument7 image3)
	(supports instrument7 infrared2)
	(calibration_target instrument7 groundstation2)
	(supports instrument8 image0)
	(supports instrument8 infrared2)
	(calibration_target instrument8 groundstation2)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument9 image3)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 infrared2)
	(supports instrument10 image0)
	(calibration_target instrument10 groundstation2)
	(supports instrument11 image3)
	(supports instrument11 image0)
	(calibration_target instrument11 groundstation2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
)
(:goal (and
	(pointing satellite0 planet4)
	(pointing satellite2 planet6)
	(have_image planet4 infrared2)
	(have_image planet5 thermograph1)
	(have_image planet6 infrared2)
))

)
