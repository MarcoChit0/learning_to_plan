; child-snack task with 1 children and 0.63 gluten factor 
; constant factor of 1.5
; random seed: 655308

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child424 - child
    bread484 - bread-portion
    content442 - content-portion
    tray57 - tray
    table126 table445 table130 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray57 kitchen)
     (at_kitchen_bread bread484)
     (at_kitchen_content content442)
     (not_allergic_gluten child424)
     (waiting child424 table126)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child424)
    )
  )
)
