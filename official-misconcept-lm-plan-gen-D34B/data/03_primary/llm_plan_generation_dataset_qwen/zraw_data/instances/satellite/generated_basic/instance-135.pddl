(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	instrument4 - instrument
	image2 - mode
	spectrograph1 - mode
	image3 - mode
	thermograph0 - mode
	star0 - direction
	groundstation1 - direction
	phenomenon2 - direction
	phenomenon3 - direction
	star4 - direction
	planet5 - direction
	star6 - direction
)
(:init
	(supports instrument0 image3)
	(supports instrument0 spectrograph1)
	(supports instrument0 image2)
	(calibration_target instrument0 star0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument1 spectrograph1)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 star0)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 groundstation1)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star6)
	(supports instrument3 spectrograph1)
	(supports instrument3 image2)
	(calibration_target instrument3 star0)
	(supports instrument4 image3)
	(calibration_target instrument4 groundstation1)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star6)
)
(:goal (and
	(pointing satellite1 star6)
	(pointing satellite2 planet5)
	(have_image phenomenon2 spectrograph1)
	(have_image phenomenon3 image3)
	(have_image star4 spectrograph1)
	(have_image planet5 image2)
	(have_image star6 image2)
))

)
