(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	spectrograph0 - mode
	groundstation1 - direction
	groundstation0 - direction
	star3 - direction
	groundstation2 - direction
	star4 - direction
	star5 - direction
	planet6 - direction
	phenomenon7 - direction
	planet8 - direction
	phenomenon9 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 star3)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star5)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 star3)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 groundstation2)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 groundstation2)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon9)
)
(:goal (and
	(pointing satellite1 groundstation1)
	(have_image star4 spectrograph0)
	(have_image star5 spectrograph0)
	(have_image planet6 spectrograph0)
	(have_image phenomenon7 spectrograph0)
	(have_image planet8 spectrograph0)
	(have_image phenomenon9 spectrograph0)
))

)
