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
	image0 - mode
	spectrograph1 - mode
	groundstation3 - direction
	groundstation2 - direction
	star1 - direction
	groundstation0 - direction
	phenomenon4 - direction
	phenomenon5 - direction
	star6 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(supports instrument0 image0)
	(calibration_target instrument0 star1)
	(supports instrument1 image0)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 spectrograph1)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument4 spectrograph1)
	(supports instrument4 image0)
	(calibration_target instrument4 star1)
	(supports instrument5 spectrograph1)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation0)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon4)
)
(:goal (and
	(pointing satellite0 groundstation3)
	(pointing satellite1 phenomenon5)
	(have_image phenomenon4 spectrograph1)
	(have_image phenomenon5 spectrograph1)
	(have_image star6 spectrograph1)
))

)
