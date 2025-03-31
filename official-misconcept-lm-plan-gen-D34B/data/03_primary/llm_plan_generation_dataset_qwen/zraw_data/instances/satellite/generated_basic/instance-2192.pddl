(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	image0 - mode
	infrared1 - mode
	groundstation1 - direction
	groundstation0 - direction
	phenomenon2 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 infrared1)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 infrared1)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 image0)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 image0)
	(supports instrument7 infrared1)
	(calibration_target instrument7 groundstation0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
)
(:goal (and
	(have_image phenomenon2 infrared1)
))

)
