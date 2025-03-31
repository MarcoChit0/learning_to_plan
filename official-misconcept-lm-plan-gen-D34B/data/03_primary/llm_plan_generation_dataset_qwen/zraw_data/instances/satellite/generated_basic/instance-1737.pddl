(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	infrared0 - mode
	spectrograph1 - mode
	thermograph2 - mode
	star0 - direction
	star2 - direction
	star4 - direction
	groundstation5 - direction
	star1 - direction
	star3 - direction
	phenomenon6 - direction
	star7 - direction
	star8 - direction
	planet9 - direction
	phenomenon10 - direction
)
(:init
	(supports instrument0 thermograph2)
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 star1)
	(calibration_target instrument0 star3)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument1 spectrograph1)
	(supports instrument1 thermograph2)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star3)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
)
(:goal (and
	(pointing satellite0 star0)
	(have_image phenomenon6 thermograph2)
	(have_image star7 thermograph2)
	(have_image star8 spectrograph1)
	(have_image planet9 spectrograph1)
	(have_image phenomenon10 infrared0)
))

)
