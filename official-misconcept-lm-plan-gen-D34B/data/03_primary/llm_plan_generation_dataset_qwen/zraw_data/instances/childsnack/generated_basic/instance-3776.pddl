; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 549271

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child306 - child
    bread149 - bread-portion
    content380 - content-portion
    tray163 - tray
    table126 table230 table100 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray163 kitchen)
     (at_kitchen_bread bread149)
     (at_kitchen_content content380)
     (not_allergic_gluten child306)
     (waiting child306 table126)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child306)
    )
  )
)
