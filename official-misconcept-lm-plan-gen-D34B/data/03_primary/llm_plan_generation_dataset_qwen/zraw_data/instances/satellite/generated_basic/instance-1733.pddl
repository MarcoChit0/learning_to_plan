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
	instrument6 - instrument
	infrared1 - mode
	image2 - mode
	thermograph0 - mode
	star3 - direction
	groundstation5 - direction
	groundstation4 - direction
	star0 - direction
	star1 - direction
	star2 - direction
	phenomenon6 - direction
	planet7 - direction
	planet8 - direction
	star9 - direction
	planet10 - direction
)
(:init
	(supports instrument0 infrared1)
	(calibration_target instrument0 groundstation4)
	(calibration_target instrument0 star3)
	(supports instrument1 thermograph0)
	(supports instrument1 image2)
	(calibration_target instrument1 star1)
	(supports instrument2 infrared1)
	(supports instrument2 image2)
	(calibration_target instrument2 groundstation5)
	(calibration_target instrument2 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation4)
	(supports instrument3 infrared1)
	(supports instrument3 thermograph0)
	(supports instrument3 image2)
	(calibration_target instrument3 groundstation4)
	(calibration_target instrument3 groundstation5)
	(supports instrument4 image2)
	(calibration_target instrument4 star0)
	(calibration_target instrument4 star2)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 star0)
	(supports instrument6 image2)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 star2)
	(calibration_target instrument6 star1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
)
(:goal (and
	(have_image phenomenon6 image2)
	(have_image planet7 thermograph0)
	(have_image planet8 image2)
	(have_image star9 infrared1)
	(have_image planet10 infrared1)
))

)
