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
	satellite3 - satellite
	instrument5 - instrument
	instrument6 - instrument
	thermograph0 - mode
	groundstation0 - direction
	groundstation2 - direction
	star3 - direction
	groundstation1 - direction
	star4 - direction
	star5 - direction
	star6 - direction
	star7 - direction
	star8 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star3)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 star4)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 groundstation2)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star7)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 star3)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 star4)
	(on_board instrument5 satellite3)
	(on_board instrument6 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star4)
)
(:goal (and
	(pointing satellite0 star4)
	(have_image star5 thermograph0)
	(have_image star6 thermograph0)
	(have_image star7 thermograph0)
	(have_image star8 thermograph0)
))

)
