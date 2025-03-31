; child-snack task with 1 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 169072

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child384 - child
    bread353 - bread-portion
    content388 - content-portion
    tray344 - tray
    table239 table121 table24 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray344 kitchen)
     (at_kitchen_bread bread353)
     (at_kitchen_content content388)
     (not_allergic_gluten child384)
     (waiting child384 table239)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child384)
    )
  )
)
