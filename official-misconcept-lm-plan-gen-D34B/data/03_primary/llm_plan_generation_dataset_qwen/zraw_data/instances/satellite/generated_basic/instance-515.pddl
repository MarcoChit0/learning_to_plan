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
	infrared3 - mode
	spectrograph2 - mode
	thermograph0 - mode
	image1 - mode
	groundstation4 - direction
	groundstation2 - direction
	star0 - direction
	groundstation3 - direction
	groundstation1 - direction
	phenomenon5 - direction
	star6 - direction
	star7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 image1)
	(supports instrument1 thermograph0)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 infrared3)
	(calibration_target instrument2 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet8)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 thermograph0)
	(supports instrument4 infrared3)
	(calibration_target instrument4 groundstation1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
)
(:goal (and
	(have_image phenomenon5 infrared3)
	(have_image star6 spectrograph2)
	(have_image star7 infrared3)
	(have_image planet8 thermograph0)
))

)
