(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	satellite2 - satellite
	instrument2 - instrument
	instrument3 - instrument
	satellite3 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	thermograph0 - mode
	groundstation4 - direction
	groundstation2 - direction
	star1 - direction
	star5 - direction
	star0 - direction
	groundstation3 - direction
	planet6 - direction
	star7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 groundstation2)
	(calibration_target instrument0 star1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star5)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation2)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star0)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 groundstation2)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation3)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 star1)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 star5)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 groundstation3)
	(calibration_target instrument6 star0)
	(on_board instrument4 satellite3)
	(on_board instrument5 satellite3)
	(on_board instrument6 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star7)
)
(:goal (and
	(pointing satellite2 groundstation4)
	(have_image planet6 thermograph0)
	(have_image star7 thermograph0)
	(have_image phenomenon8 thermograph0)
))

)
