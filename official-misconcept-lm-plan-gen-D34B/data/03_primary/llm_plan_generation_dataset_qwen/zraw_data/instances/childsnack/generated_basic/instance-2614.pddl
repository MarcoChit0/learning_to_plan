; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 116940

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child248 - child
    bread471 - bread-portion
    content253 - content-portion
    tray250 tray165 tray149 - tray
    table200 table452 table190 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray250 kitchen)
     (at tray165 kitchen)
     (at tray149 kitchen)
     (at_kitchen_bread bread471)
     (at_kitchen_content content253)
     (not_allergic_gluten child248)
     (waiting child248 table190)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child248)
    )
  )
)
