; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 985415

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child12 - child
    bread86 - bread-portion
    content103 - content-portion
    tray44 tray87 - tray
    table85 table202 table204 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray44 kitchen)
     (at tray87 kitchen)
     (at_kitchen_bread bread86)
     (at_kitchen_content content103)
     (not_allergic_gluten child12)
     (waiting child12 table202)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child12)
    )
  )
)
