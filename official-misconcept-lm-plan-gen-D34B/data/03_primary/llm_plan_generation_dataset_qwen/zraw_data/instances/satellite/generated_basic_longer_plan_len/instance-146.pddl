(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	spectrograph1 - mode
	infrared3 - mode
	image0 - mode
	image2 - mode
	star0 - direction
	groundstation1 - direction
	star4 - direction
	groundstation2 - direction
	groundstation3 - direction
	phenomenon5 - direction
	star6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
	planet9 - direction
	phenomenon10 - direction
	planet11 - direction
	phenomenon12 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(supports instrument0 image0)
	(supports instrument0 image2)
	(calibration_target instrument0 star4)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon7)
	(supports instrument1 spectrograph1)
	(supports instrument1 infrared3)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 spectrograph1)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation3)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
)
(:goal (and
	(pointing satellite0 phenomenon5)
	(have_image phenomenon5 image0)
	(have_image star6 infrared3)
	(have_image phenomenon7 image0)
	(have_image phenomenon8 image2)
	(have_image planet9 spectrograph1)
	(have_image phenomenon10 image2)
	(have_image planet11 spectrograph1)
	(have_image phenomenon12 image2)
))

)
