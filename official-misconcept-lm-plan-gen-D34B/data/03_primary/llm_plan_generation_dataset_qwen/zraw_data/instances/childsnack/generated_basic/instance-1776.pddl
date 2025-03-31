; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 209783

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child185 - child
    bread6 - bread-portion
    content381 - content-portion
    tray97 - tray
    table200 table168 table297 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray97 kitchen)
     (at_kitchen_bread bread6)
     (at_kitchen_content content381)
     (not_allergic_gluten child185)
     (waiting child185 table168)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child185)
    )
  )
)
