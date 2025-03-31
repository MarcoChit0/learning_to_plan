; child-snack task with 3 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 75684

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child426 child344 child130 - child
    bread491 bread408 bread338 - bread-portion
    content51 content284 content143 - content-portion
    tray363 tray262 tray32 - tray
    table37 table222 table175 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray363 kitchen)
     (at tray262 kitchen)
     (at tray32 kitchen)
     (at_kitchen_bread bread491)
     (at_kitchen_bread bread408)
     (at_kitchen_bread bread338)
     (at_kitchen_content content51)
     (at_kitchen_content content284)
     (at_kitchen_content content143)
     (no_gluten_bread bread338)
     (no_gluten_content content284)
     (allergic_gluten child344)
     (not_allergic_gluten child426)
     (not_allergic_gluten child130)
     (waiting child426 table37)
     (waiting child344 table175)
     (waiting child130 table175)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child426)
     (served child344)
     (served child130)
    )
  )
)
