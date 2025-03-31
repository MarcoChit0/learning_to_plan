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
	instrument8 - instrument
	image0 - mode
	image1 - mode
	thermograph2 - mode
	star0 - direction
	star4 - direction
	star3 - direction
	groundstation1 - direction
	groundstation5 - direction
	groundstation2 - direction
	phenomenon6 - direction
	phenomenon7 - direction
	star8 - direction
	phenomenon9 - direction
)
(:init
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation1)
	(calibration_target instrument0 star4)
	(supports instrument1 image1)
	(calibration_target instrument1 star0)
	(calibration_target instrument1 star3)
	(supports instrument2 image1)
	(supports instrument2 image0)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon9)
	(supports instrument3 image0)
	(supports instrument3 thermograph2)
	(supports instrument3 image1)
	(calibration_target instrument3 groundstation5)
	(calibration_target instrument3 star4)
	(supports instrument4 thermograph2)
	(supports instrument4 image1)
	(supports instrument4 image0)
	(calibration_target instrument4 star4)
	(supports instrument5 image1)
	(supports instrument5 image0)
	(calibration_target instrument5 star3)
	(supports instrument6 image1)
	(calibration_target instrument6 star3)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument7 thermograph2)
	(calibration_target instrument7 groundstation5)
	(calibration_target instrument7 groundstation1)
	(supports instrument8 image1)
	(supports instrument8 thermograph2)
	(calibration_target instrument8 groundstation2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
)
(:goal (and
	(pointing satellite2 phenomenon6)
	(have_image phenomenon6 thermograph2)
	(have_image phenomenon7 thermograph2)
	(have_image star8 image0)
	(have_image phenomenon9 image0)
))

)
