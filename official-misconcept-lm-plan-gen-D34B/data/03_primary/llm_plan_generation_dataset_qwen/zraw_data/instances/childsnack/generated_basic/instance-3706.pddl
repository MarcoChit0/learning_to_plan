; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 557763

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child455 child195 - child
    bread366 bread54 - bread-portion
    content163 content305 - content-portion
    tray382 tray233 - tray
    table268 table58 table391 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray382 kitchen)
     (at tray233 kitchen)
     (at_kitchen_bread bread366)
     (at_kitchen_bread bread54)
     (at_kitchen_content content163)
     (at_kitchen_content content305)
     (no_gluten_bread bread54)
     (no_gluten_content content305)
     (allergic_gluten child195)
     (not_allergic_gluten child455)
     (waiting child455 table58)
     (waiting child195 table58)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child455)
     (served child195)
    )
  )
)
