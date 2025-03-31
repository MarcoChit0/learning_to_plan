; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.3
; random seed: 847410

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child103 child398 - child
    bread135 bread136 - bread-portion
    content44 content6 - content-portion
    tray236 - tray
    table119 table113 table212 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray236 kitchen)
     (at_kitchen_bread bread135)
     (at_kitchen_bread bread136)
     (at_kitchen_content content44)
     (at_kitchen_content content6)
     (no_gluten_bread bread135)
     (no_gluten_content content44)
     (allergic_gluten child398)
     (not_allergic_gluten child103)
     (waiting child103 table113)
     (waiting child398 table212)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child103)
     (served child398)
    )
  )
)
