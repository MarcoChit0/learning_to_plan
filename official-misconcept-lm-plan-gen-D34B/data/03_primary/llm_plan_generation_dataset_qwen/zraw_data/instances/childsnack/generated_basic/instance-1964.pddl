; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.3
; random seed: 730577

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child74 child20 - child
    bread207 bread406 - bread-portion
    content144 content141 - content-portion
    tray401 - tray
    table289 table351 table19 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray401 kitchen)
     (at_kitchen_bread bread207)
     (at_kitchen_bread bread406)
     (at_kitchen_content content144)
     (at_kitchen_content content141)
     (no_gluten_bread bread406)
     (no_gluten_content content144)
     (allergic_gluten child20)
     (not_allergic_gluten child74)
     (waiting child74 table19)
     (waiting child20 table19)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child74)
     (served child20)
    )
  )
)
