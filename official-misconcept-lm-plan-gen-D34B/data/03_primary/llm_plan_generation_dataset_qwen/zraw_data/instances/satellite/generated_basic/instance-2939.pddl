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
	image0 - mode
	thermograph2 - mode
	infrared1 - mode
	star4 - direction
	groundstation3 - direction
	star0 - direction
	groundstation1 - direction
	groundstation2 - direction
	star5 - direction
	phenomenon6 - direction
	phenomenon7 - direction
	star8 - direction
)
(:init
	(supports instrument0 thermograph2)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 infrared1)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 infrared1)
	(supports instrument2 thermograph2)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 image0)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon6)
	(supports instrument4 image0)
	(supports instrument4 infrared1)
	(supports instrument4 thermograph2)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 thermograph2)
	(supports instrument5 infrared1)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 thermograph2)
	(supports instrument6 image0)
	(supports instrument6 infrared1)
	(calibration_target instrument6 groundstation2)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
)
(:goal (and
	(pointing satellite1 star8)
	(have_image star5 infrared1)
	(have_image phenomenon6 image0)
	(have_image phenomenon7 thermograph2)
	(have_image star8 image0)
))

)
