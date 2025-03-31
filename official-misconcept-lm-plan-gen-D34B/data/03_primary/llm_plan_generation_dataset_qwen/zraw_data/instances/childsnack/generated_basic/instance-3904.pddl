; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 40203

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child272 - child
    bread102 - bread-portion
    content0 - content-portion
    tray384 tray19 - tray
    table26 table389 table46 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray384 kitchen)
     (at tray19 kitchen)
     (at_kitchen_bread bread102)
     (at_kitchen_content content0)
     (not_allergic_gluten child272)
     (waiting child272 table389)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child272)
    )
  )
)
