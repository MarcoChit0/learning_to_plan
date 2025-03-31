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
	instrument7 - instrument
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite3 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	spectrograph1 - mode
	image2 - mode
	image0 - mode
	groundstation4 - direction
	groundstation5 - direction
	star1 - direction
	star0 - direction
	star2 - direction
	groundstation3 - direction
	phenomenon6 - direction
	phenomenon7 - direction
	star8 - direction
	star9 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 image2)
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 star0)
	(supports instrument1 image0)
	(supports instrument1 image2)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 groundstation5)
	(supports instrument2 image2)
	(supports instrument2 spectrograph1)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation5)
	(calibration_target instrument2 star2)
	(supports instrument3 image2)
	(calibration_target instrument3 star1)
	(calibration_target instrument3 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 groundstation3)
	(supports instrument5 spectrograph1)
	(supports instrument5 image2)
	(calibration_target instrument5 star2)
	(supports instrument6 spectrograph1)
	(calibration_target instrument6 star0)
	(calibration_target instrument6 star1)
	(supports instrument7 spectrograph1)
	(calibration_target instrument7 star1)
	(calibration_target instrument7 star0)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument8 image0)
	(calibration_target instrument8 groundstation5)
	(supports instrument9 image0)
	(calibration_target instrument9 star2)
	(supports instrument10 spectrograph1)
	(supports instrument10 image2)
	(calibration_target instrument10 star2)
	(calibration_target instrument10 groundstation3)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star8)
	(supports instrument11 spectrograph1)
	(supports instrument11 image0)
	(calibration_target instrument11 groundstation3)
	(calibration_target instrument11 groundstation5)
	(supports instrument12 image2)
	(calibration_target instrument12 star1)
	(calibration_target instrument12 star0)
	(supports instrument13 image2)
	(supports instrument13 image0)
	(supports instrument13 spectrograph1)
	(calibration_target instrument13 star2)
	(calibration_target instrument13 star0)
	(supports instrument14 spectrograph1)
	(supports instrument14 image0)
	(calibration_target instrument14 star0)
	(calibration_target instrument14 star1)
	(supports instrument15 spectrograph1)
	(supports instrument15 image2)
	(calibration_target instrument15 star2)
	(supports instrument16 spectrograph1)
	(supports instrument16 image0)
	(calibration_target instrument16 groundstation3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(on_board instrument16 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation5)
)
(:goal (and
	(have_image phenomenon6 image0)
	(have_image phenomenon7 image0)
	(have_image star8 image2)
	(have_image star9 spectrograph1)
))

)
