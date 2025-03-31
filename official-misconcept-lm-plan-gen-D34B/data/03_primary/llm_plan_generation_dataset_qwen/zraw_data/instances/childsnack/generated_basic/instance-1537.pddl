; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 367715

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child452 child170 - child
    bread352 bread175 - bread-portion
    content323 content498 - content-portion
    tray238 tray256 tray185 tray420 - tray
    table151 table387 table18 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray238 kitchen)
     (at tray256 kitchen)
     (at tray185 kitchen)
     (at tray420 kitchen)
     (at_kitchen_bread bread352)
     (at_kitchen_bread bread175)
     (at_kitchen_content content323)
     (at_kitchen_content content498)
     (not_allergic_gluten child170)
     (not_allergic_gluten child452)
     (waiting child452 table151)
     (waiting child170 table18)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child452)
     (served child170)
    )
  )
)
