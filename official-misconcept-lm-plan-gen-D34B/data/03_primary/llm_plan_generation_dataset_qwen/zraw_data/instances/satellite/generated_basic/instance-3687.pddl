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
	satellite2 - satellite
	instrument6 - instrument
	thermograph1 - mode
	image0 - mode
	groundstation1 - direction
	star5 - direction
	star2 - direction
	groundstation3 - direction
	star0 - direction
	groundstation4 - direction
	planet6 - direction
	phenomenon7 - direction
	planet8 - direction
	star9 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 thermograph1)
	(calibration_target instrument0 groundstation3)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 image0)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 groundstation1)
	(calibration_target instrument1 star5)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation4)
	(calibration_target instrument2 star2)
	(supports instrument3 thermograph1)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation1)
	(calibration_target instrument3 star5)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star9)
	(supports instrument4 thermograph1)
	(supports instrument4 image0)
	(calibration_target instrument4 star2)
	(calibration_target instrument4 star5)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 star0)
	(calibration_target instrument5 groundstation3)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star5)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 groundstation4)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet8)
)
(:goal (and
	(pointing satellite2 groundstation3)
	(have_image planet6 thermograph1)
	(have_image phenomenon7 image0)
	(have_image planet8 thermograph1)
	(have_image star9 image0)
))

)
