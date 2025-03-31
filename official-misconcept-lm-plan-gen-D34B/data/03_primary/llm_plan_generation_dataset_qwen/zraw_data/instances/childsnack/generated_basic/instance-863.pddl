; child-snack task with 2 children and 0.59 gluten factor 
; constant factor of 1.9
; random seed: 825850

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child176 child98 - child
    bread272 bread67 - bread-portion
    content357 content262 - content-portion
    tray310 tray285 - tray
    table213 table341 table112 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray310 kitchen)
     (at tray285 kitchen)
     (at_kitchen_bread bread272)
     (at_kitchen_bread bread67)
     (at_kitchen_content content357)
     (at_kitchen_content content262)
     (no_gluten_bread bread67)
     (no_gluten_content content262)
     (allergic_gluten child176)
     (not_allergic_gluten child98)
     (waiting child176 table213)
     (waiting child98 table213)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child176)
     (served child98)
    )
  )
)
