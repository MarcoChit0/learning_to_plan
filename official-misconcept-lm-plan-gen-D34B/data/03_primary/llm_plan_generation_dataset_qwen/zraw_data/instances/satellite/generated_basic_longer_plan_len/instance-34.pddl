(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	satellite2 - satellite
	instrument2 - instrument
	satellite3 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	image2 - mode
	thermograph3 - mode
	thermograph1 - mode
	image0 - mode
	groundstation2 - direction
	star1 - direction
	groundstation0 - direction
	groundstation3 - direction
	star4 - direction
	phenomenon5 - direction
	planet6 - direction
	planet7 - direction
	planet8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet7)
	(supports instrument1 thermograph3)
	(calibration_target instrument1 groundstation3)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument2 image2)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 groundstation2)
	(on_board instrument2 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
	(supports instrument3 thermograph3)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 image0)
	(supports instrument4 thermograph3)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 star1)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 thermograph3)
	(supports instrument6 image2)
	(calibration_target instrument6 groundstation3)
	(on_board instrument3 satellite3)
	(on_board instrument4 satellite3)
	(on_board instrument5 satellite3)
	(on_board instrument6 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet6)
)
(:goal (and
	(have_image star4 thermograph3)
	(have_image phenomenon5 image0)
	(have_image planet6 image2)
	(have_image planet7 image2)
	(have_image planet8 thermograph3)
	(have_image planet9 thermograph3)
))

)
