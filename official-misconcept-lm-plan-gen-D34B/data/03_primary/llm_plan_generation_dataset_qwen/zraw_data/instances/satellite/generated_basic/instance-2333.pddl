(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	spectrograph0 - mode
	thermograph1 - mode
	infrared2 - mode
	star0 - direction
	star1 - direction
	groundstation3 - direction
	star2 - direction
	groundstation4 - direction
	phenomenon5 - direction
	planet6 - direction
	phenomenon7 - direction
	star8 - direction
	star9 - direction
)
(:init
	(supports instrument0 infrared2)
	(calibration_target instrument0 groundstation4)
	(supports instrument1 infrared2)
	(supports instrument1 thermograph1)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation4)
	(supports instrument2 infrared2)
	(calibration_target instrument2 groundstation4)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star9)
)
(:goal (and
	(have_image phenomenon5 thermograph1)
	(have_image planet6 spectrograph0)
	(have_image phenomenon7 spectrograph0)
	(have_image star8 thermograph1)
	(have_image star9 infrared2)
))

)
