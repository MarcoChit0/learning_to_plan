(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	infrared2 - mode
	thermograph1 - mode
	spectrograph3 - mode
	image0 - mode
	star0 - direction
	groundstation1 - direction
	groundstation3 - direction
	groundstation2 - direction
	groundstation4 - direction
	phenomenon5 - direction
	planet6 - direction
	planet7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 infrared2)
	(supports instrument0 spectrograph3)
	(supports instrument0 thermograph1)
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation2)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon5)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 groundstation4)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
)
(:goal (and
	(pointing satellite1 groundstation2)
	(have_image phenomenon5 image0)
	(have_image planet6 infrared2)
	(have_image planet7 image0)
	(have_image phenomenon8 image0)
))

)
