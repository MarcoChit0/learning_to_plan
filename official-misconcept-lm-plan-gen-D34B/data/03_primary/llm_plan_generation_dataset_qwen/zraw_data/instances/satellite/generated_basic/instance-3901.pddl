(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	image0 - mode
	spectrograph1 - mode
	infrared3 - mode
	infrared2 - mode
	star0 - direction
	star1 - direction
	planet2 - direction
	phenomenon3 - direction
)
(:init
	(supports instrument0 infrared3)
	(calibration_target instrument0 star1)
	(supports instrument1 image0)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument2 spectrograph1)
	(supports instrument2 infrared2)
	(supports instrument2 infrared3)
	(calibration_target instrument2 star0)
	(supports instrument3 image0)
	(supports instrument3 infrared2)
	(calibration_target instrument3 star1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon3)
)
(:goal (and
	(pointing satellite0 phenomenon3)
	(pointing satellite1 star0)
	(have_image planet2 image0)
	(have_image phenomenon3 image0)
))

)
