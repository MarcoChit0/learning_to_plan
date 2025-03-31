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
	satellite3 - satellite
	instrument6 - instrument
	instrument7 - instrument
	infrared1 - mode
	thermograph2 - mode
	image3 - mode
	thermograph0 - mode
	star0 - direction
	star2 - direction
	groundstation3 - direction
	groundstation4 - direction
	groundstation1 - direction
	star5 - direction
	star6 - direction
)
(:init
	(supports instrument0 thermograph0)
	(supports instrument0 image3)
	(supports instrument0 thermograph2)
	(calibration_target instrument0 groundstation4)
	(supports instrument1 thermograph0)
	(supports instrument1 infrared1)
	(supports instrument1 image3)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 infrared1)
	(supports instrument2 thermograph2)
	(supports instrument2 image3)
	(calibration_target instrument2 groundstation4)
	(supports instrument3 image3)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation4)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star6)
	(supports instrument5 thermograph0)
	(supports instrument5 infrared1)
	(calibration_target instrument5 groundstation1)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star6)
	(supports instrument6 infrared1)
	(supports instrument6 thermograph0)
	(supports instrument6 thermograph2)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 infrared1)
	(supports instrument7 image3)
	(calibration_target instrument7 groundstation1)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star6)
)
(:goal (and
	(pointing satellite0 star0)
	(pointing satellite2 star5)
	(pointing satellite3 star5)
	(have_image star5 image3)
	(have_image star6 thermograph2)
))

)
