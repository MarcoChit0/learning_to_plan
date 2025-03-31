(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	spectrograph1 - mode
	thermograph3 - mode
	infrared2 - mode
	infrared0 - mode
	star1 - direction
	groundstation2 - direction
	groundstation4 - direction
	star3 - direction
	star0 - direction
	planet5 - direction
	planet6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 infrared2)
	(calibration_target instrument0 groundstation4)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument1 spectrograph1)
	(supports instrument1 infrared0)
	(supports instrument1 infrared2)
	(calibration_target instrument1 star3)
	(supports instrument2 infrared2)
	(supports instrument2 spectrograph1)
	(supports instrument2 thermograph3)
	(calibration_target instrument2 star0)
	(supports instrument3 infrared2)
	(calibration_target instrument3 star0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation4)
)
(:goal (and
	(pointing satellite0 phenomenon8)
	(pointing satellite1 star1)
	(have_image planet5 thermograph3)
	(have_image planet6 spectrograph1)
	(have_image phenomenon7 infrared2)
	(have_image phenomenon8 infrared2)
))

)
