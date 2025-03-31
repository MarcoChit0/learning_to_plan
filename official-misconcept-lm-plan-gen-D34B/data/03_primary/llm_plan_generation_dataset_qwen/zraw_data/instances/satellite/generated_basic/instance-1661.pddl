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
	instrument7 - instrument
	image0 - mode
	groundstation3 - direction
	groundstation2 - direction
	groundstation4 - direction
	star1 - direction
	groundstation5 - direction
	star0 - direction
	planet6 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star0)
	(calibration_target instrument0 groundstation4)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation5)
	(supports instrument1 image0)
	(calibration_target instrument1 star0)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation4)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation3)
	(calibration_target instrument2 groundstation4)
	(on_board instrument2 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet6)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation5)
	(calibration_target instrument4 groundstation4)
	(supports instrument5 image0)
	(calibration_target instrument5 star0)
	(calibration_target instrument5 star1)
	(supports instrument6 image0)
	(calibration_target instrument6 star0)
	(calibration_target instrument6 groundstation5)
	(supports instrument7 image0)
	(calibration_target instrument7 star0)
	(on_board instrument3 satellite3)
	(on_board instrument4 satellite3)
	(on_board instrument5 satellite3)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation3)
)
(:goal (and
	(pointing satellite1 groundstation4)
	(have_image planet6 image0)
))

)
