; child-snack task with 1 children and 0.63 gluten factor 
; constant factor of 1.5
; random seed: 422652

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child420 - child
    bread276 - bread-portion
    content93 - content-portion
    tray354 - tray
    table261 table323 table129 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray354 kitchen)
     (at_kitchen_bread bread276)
     (at_kitchen_content content93)
     (not_allergic_gluten child420)
     (waiting child420 table261)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child420)
    )
  )
)
