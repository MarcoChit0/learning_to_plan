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
	instrument6 - instrument
	spectrograph2 - mode
	spectrograph0 - mode
	image3 - mode
	image1 - mode
	groundstation2 - direction
	star1 - direction
	groundstation3 - direction
	groundstation0 - direction
	star4 - direction
	planet5 - direction
	planet6 - direction
	star7 - direction
	planet8 - direction
	phenomenon9 - direction
	star10 - direction
	planet11 - direction
	phenomenon12 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 star1)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star4)
	(supports instrument2 image3)
	(supports instrument2 spectrograph0)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 star4)
	(supports instrument3 image1)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument4 image1)
	(supports instrument4 spectrograph0)
	(supports instrument4 spectrograph2)
	(calibration_target instrument4 groundstation3)
	(supports instrument5 spectrograph2)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 star4)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet8)
)
(:goal (and
	(pointing satellite0 planet11)
	(pointing satellite1 planet8)
	(have_image planet5 image3)
	(have_image planet6 spectrograph0)
	(have_image star7 spectrograph2)
	(have_image planet8 spectrograph0)
	(have_image phenomenon9 image3)
	(have_image star10 image3)
	(have_image planet11 image3)
	(have_image phenomenon12 spectrograph2)
))

)
