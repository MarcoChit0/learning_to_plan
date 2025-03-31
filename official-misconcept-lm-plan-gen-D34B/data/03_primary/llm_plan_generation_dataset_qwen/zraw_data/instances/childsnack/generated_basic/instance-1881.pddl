; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.7
; random seed: 15656

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child176 child224 - child
    bread222 bread6 - bread-portion
    content281 content260 - content-portion
    tray9 tray361 - tray
    table68 table75 table149 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray9 kitchen)
     (at tray361 kitchen)
     (at_kitchen_bread bread222)
     (at_kitchen_bread bread6)
     (at_kitchen_content content281)
     (at_kitchen_content content260)
     (no_gluten_bread bread6)
     (no_gluten_content content281)
     (allergic_gluten child176)
     (not_allergic_gluten child224)
     (waiting child176 table149)
     (waiting child224 table68)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child176)
     (served child224)
    )
  )
)
