(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	image2 - mode
	thermograph0 - mode
	infrared1 - mode
	groundstation1 - direction
	star2 - direction
	star3 - direction
	groundstation0 - direction
	groundstation4 - direction
	groundstation5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 image2)
	(calibration_target instrument0 star2)
	(calibration_target instrument0 groundstation5)
	(supports instrument1 image2)
	(calibration_target instrument1 groundstation0)
	(calibration_target instrument1 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument2 image2)
	(supports instrument2 infrared1)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star2)
	(calibration_target instrument2 groundstation5)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon6)
	(supports instrument3 infrared1)
	(supports instrument3 thermograph0)
	(supports instrument3 image2)
	(calibration_target instrument3 star3)
	(supports instrument4 image2)
	(calibration_target instrument4 star3)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 thermograph0)
	(supports instrument5 image2)
	(supports instrument5 infrared1)
	(calibration_target instrument5 groundstation5)
	(calibration_target instrument5 groundstation4)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 groundstation4)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 infrared1)
	(calibration_target instrument7 groundstation5)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
)
(:goal (and
	(pointing satellite1 phenomenon6)
	(have_image phenomenon6 thermograph0)
))

)
