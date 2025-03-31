(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	thermograph1 - mode
	thermograph0 - mode
	spectrograph2 - mode
	groundstation3 - direction
	groundstation4 - direction
	star0 - direction
	groundstation5 - direction
	groundstation1 - direction
	groundstation2 - direction
	phenomenon6 - direction
	planet7 - direction
	star8 - direction
	star9 - direction
	planet10 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 spectrograph2)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 groundstation5)
	(calibration_target instrument0 star0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation2)
	(calibration_target instrument1 groundstation1)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star9)
)
(:goal (and
	(pointing satellite1 star9)
	(have_image phenomenon6 thermograph1)
	(have_image planet7 thermograph0)
	(have_image star8 thermograph0)
	(have_image star9 thermograph1)
	(have_image planet10 thermograph1)
))

)
