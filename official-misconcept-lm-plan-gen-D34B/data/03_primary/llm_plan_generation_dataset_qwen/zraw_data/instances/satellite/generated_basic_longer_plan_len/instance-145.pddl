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
	instrument10 - instrument
	instrument11 - instrument
	image2 - mode
	image3 - mode
	thermograph1 - mode
	image0 - mode
	star4 - direction
	groundstation0 - direction
	star3 - direction
	groundstation2 - direction
	groundstation1 - direction
	star5 - direction
	star6 - direction
	star7 - direction
	phenomenon8 - direction
	star9 - direction
	planet10 - direction
	phenomenon11 - direction
	phenomenon12 - direction
)
(:init
	(supports instrument0 image2)
	(supports instrument0 image0)
	(calibration_target instrument0 star4)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 image0)
	(supports instrument2 image3)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 star3)
	(supports instrument3 image0)
	(supports instrument3 image3)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 image0)
	(calibration_target instrument4 star3)
	(supports instrument5 image3)
	(supports instrument5 image0)
	(supports instrument5 image2)
	(calibration_target instrument5 star4)
	(supports instrument6 image0)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 star4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(on_board instrument6 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon12)
	(supports instrument7 image2)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 star4)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 thermograph1)
	(supports instrument9 image2)
	(supports instrument9 image0)
	(calibration_target instrument9 star3)
	(supports instrument10 image3)
	(supports instrument10 thermograph1)
	(supports instrument10 image0)
	(calibration_target instrument10 groundstation2)
	(supports instrument11 image2)
	(supports instrument11 image0)
	(calibration_target instrument11 groundstation1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(on_board instrument10 satellite1)
	(on_board instrument11 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
)
(:goal (and
	(have_image star5 image0)
	(have_image star6 image3)
	(have_image star7 image0)
	(have_image phenomenon8 image2)
	(have_image star9 image0)
	(have_image planet10 image2)
	(have_image phenomenon11 thermograph1)
	(have_image phenomenon12 image0)
))

)
