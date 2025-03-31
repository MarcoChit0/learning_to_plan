; child-snack task with 2 children and 0.59 gluten factor 
; constant factor of 1.9
; random seed: 371748

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child284 child214 - child
    bread148 bread198 - bread-portion
    content6 content147 - content-portion
    tray5 tray101 - tray
    table149 table412 table424 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray5 kitchen)
     (at tray101 kitchen)
     (at_kitchen_bread bread148)
     (at_kitchen_bread bread198)
     (at_kitchen_content content6)
     (at_kitchen_content content147)
     (no_gluten_bread bread148)
     (no_gluten_content content6)
     (allergic_gluten child214)
     (not_allergic_gluten child284)
     (waiting child284 table149)
     (waiting child214 table412)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child284)
     (served child214)
    )
  )
)
