; child-snack task with 1 children and 0.7 gluten factor 
; constant factor of 1.3
; random seed: 586223

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child183 - child
    bread353 - bread-portion
    content434 - content-portion
    tray164 - tray
    table34 table108 table242 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray164 kitchen)
     (at_kitchen_bread bread353)
     (at_kitchen_content content434)
     (not_allergic_gluten child183)
     (waiting child183 table242)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child183)
    )
  )
)
