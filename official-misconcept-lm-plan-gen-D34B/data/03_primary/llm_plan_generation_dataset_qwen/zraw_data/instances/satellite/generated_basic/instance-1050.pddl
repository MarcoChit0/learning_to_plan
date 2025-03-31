(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	infrared0 - mode
	thermograph2 - mode
	image1 - mode
	star0 - direction
	groundstation3 - direction
	groundstation1 - direction
	star2 - direction
	planet4 - direction
	phenomenon5 - direction
	star6 - direction
	planet7 - direction
	star8 - direction
)
(:init
	(supports instrument0 thermograph2)
	(supports instrument0 image1)
	(calibration_target instrument0 star0)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument2 image1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star2)
	(supports instrument3 infrared0)
	(supports instrument3 thermograph2)
	(supports instrument3 image1)
	(calibration_target instrument3 star2)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation3)
	(supports instrument5 infrared0)
	(supports instrument5 thermograph2)
	(supports instrument5 image1)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 thermograph2)
	(calibration_target instrument6 star2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
)
(:goal (and
	(pointing satellite0 star0)
	(pointing satellite1 groundstation3)
	(pointing satellite2 star0)
	(have_image planet4 infrared0)
	(have_image phenomenon5 thermograph2)
	(have_image star6 image1)
	(have_image planet7 image1)
	(have_image star8 infrared0)
))

)
