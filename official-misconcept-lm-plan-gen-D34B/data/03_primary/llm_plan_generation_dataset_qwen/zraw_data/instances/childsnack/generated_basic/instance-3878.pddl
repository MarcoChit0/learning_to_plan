; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 313231

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child102 - child
    bread210 - bread-portion
    content242 - content-portion
    tray170 tray72 - tray
    table133 table248 table111 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray170 kitchen)
     (at tray72 kitchen)
     (at_kitchen_bread bread210)
     (at_kitchen_content content242)
     (not_allergic_gluten child102)
     (waiting child102 table111)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child102)
    )
  )
)
