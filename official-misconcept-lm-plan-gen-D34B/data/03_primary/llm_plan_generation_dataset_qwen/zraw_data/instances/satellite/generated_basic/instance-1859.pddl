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
	infrared1 - mode
	thermograph0 - mode
	star1 - direction
	groundstation2 - direction
	groundstation0 - direction
	planet3 - direction
	star4 - direction
	planet5 - direction
	star6 - direction
	star7 - direction
)
(:init
	(supports instrument0 thermograph0)
	(supports instrument0 infrared1)
	(calibration_target instrument0 star1)
	(supports instrument1 thermograph0)
	(supports instrument1 infrared1)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 thermograph0)
	(supports instrument3 infrared1)
	(calibration_target instrument3 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument4 infrared1)
	(calibration_target instrument4 groundstation2)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star7)
	(supports instrument5 infrared1)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 infrared1)
	(calibration_target instrument6 groundstation0)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star6)
)
(:goal (and
	(pointing satellite0 planet5)
	(have_image planet3 infrared1)
	(have_image star4 infrared1)
	(have_image planet5 infrared1)
	(have_image star6 infrared1)
	(have_image star7 thermograph0)
))

)
