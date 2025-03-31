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
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	image2 - mode
	infrared1 - mode
	thermograph3 - mode
	image0 - mode
	star2 - direction
	star1 - direction
	groundstation0 - direction
	planet3 - direction
	star4 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 image2)
	(supports instrument0 thermograph3)
	(calibration_target instrument0 star2)
	(supports instrument1 thermograph3)
	(calibration_target instrument1 star2)
	(supports instrument2 thermograph3)
	(supports instrument2 image2)
	(calibration_target instrument2 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument3 image2)
	(calibration_target instrument3 star1)
	(supports instrument4 image0)
	(supports instrument4 infrared1)
	(calibration_target instrument4 star2)
	(supports instrument5 image0)
	(supports instrument5 thermograph3)
	(calibration_target instrument5 groundstation0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet3)
	(supports instrument6 infrared1)
	(supports instrument6 image2)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 thermograph3)
	(supports instrument7 image0)
	(supports instrument7 infrared1)
	(calibration_target instrument7 star1)
	(supports instrument8 infrared1)
	(calibration_target instrument8 groundstation0)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
)
(:goal (and
	(have_image planet3 thermograph3)
	(have_image star4 image0)
))

)
