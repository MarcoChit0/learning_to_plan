(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	image1 - mode
	thermograph0 - mode
	image2 - mode
	groundstation0 - direction
	star2 - direction
	groundstation1 - direction
	groundstation4 - direction
	groundstation3 - direction
	groundstation5 - direction
	star6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 image2)
	(supports instrument0 image1)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 groundstation5)
	(calibration_target instrument0 star2)
	(supports instrument1 image1)
	(calibration_target instrument1 star2)
	(supports instrument2 image1)
	(supports instrument2 image2)
	(calibration_target instrument2 groundstation1)
	(calibration_target instrument2 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation4)
	(supports instrument3 image2)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 groundstation4)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon8)
	(supports instrument4 image2)
	(supports instrument4 thermograph0)
	(supports instrument4 image1)
	(calibration_target instrument4 groundstation1)
	(calibration_target instrument4 groundstation3)
	(supports instrument5 image1)
	(supports instrument5 image2)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 groundstation4)
	(supports instrument6 image1)
	(supports instrument6 image2)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 groundstation5)
	(calibration_target instrument6 groundstation3)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
)
(:goal (and
	(have_image star6 image2)
	(have_image phenomenon7 image1)
	(have_image phenomenon8 image2)
))

)
