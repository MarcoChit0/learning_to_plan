; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 586349

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child401 - child
    bread354 - bread-portion
    content224 - content-portion
    tray446 tray58 tray185 tray117 - tray
    table121 table410 table117 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray446 kitchen)
     (at tray58 kitchen)
     (at tray185 kitchen)
     (at tray117 kitchen)
     (at_kitchen_bread bread354)
     (at_kitchen_content content224)
     (not_allergic_gluten child401)
     (waiting child401 table121)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child401)
    )
  )
)
