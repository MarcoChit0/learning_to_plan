; child-snack task with 1 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 284008

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child43 - child
    bread320 - bread-portion
    content449 - content-portion
    tray6 tray141 - tray
    table239 table289 table389 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray6 kitchen)
     (at tray141 kitchen)
     (at_kitchen_bread bread320)
     (at_kitchen_content content449)
     (not_allergic_gluten child43)
     (waiting child43 table289)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child43)
    )
  )
)
