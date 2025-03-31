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
	spectrograph2 - mode
	image0 - mode
	infrared3 - mode
	infrared1 - mode
	groundstation1 - direction
	groundstation0 - direction
	planet2 - direction
)
(:init
	(supports instrument0 spectrograph2)
	(supports instrument0 infrared3)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 infrared1)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 infrared1)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 infrared1)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument4 image0)
	(supports instrument4 infrared1)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 spectrograph2)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation0)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
)
(:goal (and
	(pointing satellite1 groundstation1)
	(have_image planet2 infrared1)
))

)
