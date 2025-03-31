(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	satellite3 - satellite
	instrument5 - instrument
	instrument6 - instrument
	spectrograph1 - mode
	spectrograph2 - mode
	image3 - mode
	spectrograph0 - mode
	star0 - direction
	groundstation2 - direction
	star3 - direction
	groundstation4 - direction
	star1 - direction
	star5 - direction
	star6 - direction
	phenomenon7 - direction
	star8 - direction
)
(:init
	(supports instrument0 image3)
	(supports instrument0 spectrograph0)
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 star1)
	(supports instrument1 image3)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 groundstation4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon7)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 groundstation4)
	(supports instrument3 spectrograph0)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 groundstation4)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star8)
	(supports instrument4 spectrograph2)
	(calibration_target instrument4 groundstation4)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
	(supports instrument5 spectrograph2)
	(supports instrument5 image3)
	(calibration_target instrument5 star1)
	(supports instrument6 spectrograph1)
	(supports instrument6 spectrograph2)
	(calibration_target instrument6 star1)
	(on_board instrument5 satellite3)
	(on_board instrument6 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star1)
)
(:goal (and
	(pointing satellite3 star5)
	(have_image star5 spectrograph0)
	(have_image star6 spectrograph1)
	(have_image phenomenon7 spectrograph0)
	(have_image star8 spectrograph1)
))

)
