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
	instrument5 - instrument
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	satellite4 - satellite
	instrument9 - instrument
	instrument10 - instrument
	infrared3 - mode
	infrared0 - mode
	image2 - mode
	thermograph1 - mode
	groundstation1 - direction
	star4 - direction
	groundstation0 - direction
	groundstation2 - direction
	star3 - direction
	planet5 - direction
	phenomenon6 - direction
	planet7 - direction
	star8 - direction
	star9 - direction
	planet10 - direction
)
(:init
	(supports instrument0 image2)
	(supports instrument0 infrared0)
	(supports instrument0 thermograph1)
	(calibration_target instrument0 star3)
	(supports instrument1 infrared3)
	(calibration_target instrument1 star4)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument3 thermograph1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star3)
	(supports instrument4 infrared0)
	(supports instrument4 image2)
	(calibration_target instrument4 star4)
	(supports instrument5 infrared3)
	(supports instrument5 infrared0)
	(calibration_target instrument5 star4)
	(supports instrument6 thermograph1)
	(supports instrument6 infrared3)
	(calibration_target instrument6 star4)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet5)
	(supports instrument7 image2)
	(calibration_target instrument7 groundstation0)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet10)
	(supports instrument8 image2)
	(supports instrument8 thermograph1)
	(supports instrument8 infrared0)
	(calibration_target instrument8 groundstation2)
	(on_board instrument8 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star4)
	(supports instrument9 image2)
	(supports instrument9 thermograph1)
	(supports instrument9 infrared0)
	(calibration_target instrument9 groundstation2)
	(supports instrument10 image2)
	(supports instrument10 infrared0)
	(supports instrument10 infrared3)
	(calibration_target instrument10 star3)
	(on_board instrument9 satellite4)
	(on_board instrument10 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet5)
)
(:goal (and
	(pointing satellite3 planet7)
	(have_image planet5 infrared3)
	(have_image phenomenon6 infrared3)
	(have_image planet7 infrared3)
	(have_image star8 thermograph1)
	(have_image star9 image2)
	(have_image planet10 image2)
))

)
