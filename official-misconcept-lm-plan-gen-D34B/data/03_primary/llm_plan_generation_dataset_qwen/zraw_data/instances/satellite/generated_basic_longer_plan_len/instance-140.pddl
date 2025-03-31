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
	instrument6 - instrument
	satellite1 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	thermograph1 - mode
	image2 - mode
	infrared0 - mode
	infrared3 - mode
	groundstation2 - direction
	star3 - direction
	star0 - direction
	groundstation4 - direction
	star1 - direction
	planet5 - direction
	star6 - direction
	phenomenon7 - direction
	star8 - direction
	phenomenon9 - direction
	phenomenon10 - direction
	phenomenon11 - direction
	planet12 - direction
)
(:init
	(supports instrument0 infrared3)
	(calibration_target instrument0 star3)
	(supports instrument1 infrared3)
	(calibration_target instrument1 star1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 thermograph1)
	(supports instrument4 image2)
	(calibration_target instrument4 groundstation4)
	(supports instrument5 infrared3)
	(supports instrument5 infrared0)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 star3)
	(supports instrument6 image2)
	(supports instrument6 infrared3)
	(supports instrument6 infrared0)
	(calibration_target instrument6 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(on_board instrument6 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon7)
	(supports instrument7 image2)
	(supports instrument7 thermograph1)
	(supports instrument7 infrared3)
	(calibration_target instrument7 star1)
	(supports instrument8 infrared3)
	(supports instrument8 infrared0)
	(supports instrument8 image2)
	(calibration_target instrument8 groundstation4)
	(supports instrument9 infrared0)
	(supports instrument9 infrared3)
	(supports instrument9 image2)
	(calibration_target instrument9 star1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star8)
)
(:goal (and
	(have_image planet5 infrared0)
	(have_image star6 image2)
	(have_image phenomenon7 thermograph1)
	(have_image star8 infrared0)
	(have_image phenomenon9 thermograph1)
	(have_image phenomenon10 infrared0)
	(have_image phenomenon11 image2)
	(have_image planet12 thermograph1)
))

)
