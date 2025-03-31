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
	infrared2 - mode
	image0 - mode
	thermograph3 - mode
	star3 - direction
	star5 - direction
	groundstation4 - direction
	star1 - direction
	star2 - direction
	groundstation0 - direction
	planet6 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 thermograph3)
	(calibration_target instrument0 groundstation0)
	(calibration_target instrument0 star1)
	(supports instrument1 image1)
	(supports instrument1 thermograph3)
	(supports instrument1 image0)
	(calibration_target instrument1 star2)
	(supports instrument2 infrared2)
	(supports instrument2 image1)
	(supports instrument2 thermograph3)
	(calibration_target instrument2 groundstation4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument3 image0)
	(supports instrument3 infrared2)
	(calibration_target instrument3 groundstation4)
	(calibration_target instrument3 star1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument4 image0)
	(supports instrument4 infrared2)
	(calibration_target instrument4 star2)
	(calibration_target instrument4 star1)
	(supports instrument5 infrared2)
	(supports instrument5 thermograph3)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 thermograph3)
	(supports instrument6 image1)
	(supports instrument6 image0)
	(calibration_target instrument6 groundstation0)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation4)
)
(:goal (and
	(pointing satellite0 groundstation4)
	(pointing satellite2 star5)
	(have_image planet6 image0)
))

)
