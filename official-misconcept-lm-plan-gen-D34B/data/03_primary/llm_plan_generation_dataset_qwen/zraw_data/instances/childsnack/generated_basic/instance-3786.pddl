; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 593888

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child29 - child
    bread202 - bread-portion
    content31 - content-portion
    tray130 - tray
    table3 table203 table145 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray130 kitchen)
     (at_kitchen_bread bread202)
     (at_kitchen_content content31)
     (not_allergic_gluten child29)
     (waiting child29 table145)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child29)
    )
  )
)
