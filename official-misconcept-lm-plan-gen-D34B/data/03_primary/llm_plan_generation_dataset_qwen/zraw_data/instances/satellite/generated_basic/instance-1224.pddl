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
	infrared3 - mode
	thermograph1 - mode
	spectrograph0 - mode
	image2 - mode
	groundstation1 - direction
	star0 - direction
	star2 - direction
	planet3 - direction
	phenomenon4 - direction
	planet5 - direction
)
(:init
	(supports instrument0 image2)
	(supports instrument0 thermograph1)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star0)
	(supports instrument2 spectrograph0)
	(supports instrument2 image2)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet5)
	(supports instrument4 image2)
	(calibration_target instrument4 star2)
	(supports instrument5 image2)
	(supports instrument5 infrared3)
	(calibration_target instrument5 star2)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
)
(:goal (and
	(pointing satellite0 groundstation1)
	(have_image planet3 spectrograph0)
	(have_image phenomenon4 infrared3)
	(have_image planet5 image2)
))

)
