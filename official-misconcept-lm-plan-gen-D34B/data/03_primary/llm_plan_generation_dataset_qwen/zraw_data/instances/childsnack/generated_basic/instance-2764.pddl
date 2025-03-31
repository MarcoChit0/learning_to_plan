; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 833424

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child316 child74 - child
    bread437 bread388 - bread-portion
    content369 content98 - content-portion
    tray341 - tray
    table349 table450 table190 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray341 kitchen)
     (at_kitchen_bread bread437)
     (at_kitchen_bread bread388)
     (at_kitchen_content content369)
     (at_kitchen_content content98)
     (no_gluten_bread bread437)
     (no_gluten_content content369)
     (allergic_gluten child74)
     (not_allergic_gluten child316)
     (waiting child316 table190)
     (waiting child74 table190)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child316)
     (served child74)
    )
  )
)
