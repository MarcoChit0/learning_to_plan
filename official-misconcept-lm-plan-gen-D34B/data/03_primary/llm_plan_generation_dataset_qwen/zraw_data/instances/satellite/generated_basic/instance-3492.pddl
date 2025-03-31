(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	image2 - mode
	thermograph1 - mode
	infrared0 - mode
	groundstation0 - direction
	groundstation1 - direction
	phenomenon2 - direction
	planet3 - direction
)
(:init
	(supports instrument0 image2)
	(supports instrument0 thermograph1)
	(calibration_target instrument0 groundstation0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon2)
	(supports instrument1 image2)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 image2)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 infrared0)
	(supports instrument3 image2)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 thermograph1)
	(supports instrument4 infrared0)
	(supports instrument4 image2)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 infrared0)
	(supports instrument5 image2)
	(calibration_target instrument5 groundstation0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 infrared0)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 image2)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 groundstation0)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
	(supports instrument9 image2)
	(calibration_target instrument9 groundstation1)
	(supports instrument10 infrared0)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 image2)
	(supports instrument11 thermograph1)
	(supports instrument11 infrared0)
	(calibration_target instrument11 groundstation1)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon2)
)
(:goal (and
	(pointing satellite2 groundstation0)
	(pointing satellite3 groundstation1)
	(have_image phenomenon2 image2)
	(have_image planet3 infrared0)
))

)
