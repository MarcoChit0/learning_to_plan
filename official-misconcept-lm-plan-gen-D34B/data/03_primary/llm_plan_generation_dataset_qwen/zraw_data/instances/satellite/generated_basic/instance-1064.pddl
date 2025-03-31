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
	thermograph1 - mode
	thermograph0 - mode
	image2 - mode
	star0 - direction
	groundstation2 - direction
	groundstation3 - direction
	star1 - direction
	groundstation4 - direction
	star5 - direction
	star6 - direction
	planet7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 image2)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 image2)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 image2)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 star1)
	(calibration_target instrument2 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet7)
	(supports instrument3 thermograph0)
	(supports instrument3 image2)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation4)
	(calibration_target instrument4 star5)
	(supports instrument5 thermograph0)
	(supports instrument5 image2)
	(calibration_target instrument5 star5)
	(calibration_target instrument5 star1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 star5)
	(calibration_target instrument6 groundstation4)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet7)
)
(:goal (and
	(pointing satellite1 star6)
	(pointing satellite2 phenomenon8)
	(have_image star6 thermograph1)
	(have_image planet7 image2)
	(have_image phenomenon8 thermograph1)
))

)
