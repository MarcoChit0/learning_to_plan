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
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	thermograph2 - mode
	image1 - mode
	image0 - mode
	infrared3 - mode
	groundstation0 - direction
	star1 - direction
	star2 - direction
	planet3 - direction
)
(:init
	(supports instrument0 image1)
	(calibration_target instrument0 star1)
	(supports instrument1 image1)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 thermograph2)
	(supports instrument2 image1)
	(supports instrument2 infrared3)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 thermograph2)
	(supports instrument3 image1)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument4 image0)
	(calibration_target instrument4 star1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument5 image1)
	(supports instrument5 thermograph2)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 thermograph2)
	(calibration_target instrument6 star1)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
)
(:goal (and
	(have_image star2 image0)
	(have_image planet3 image0)
))

)
