; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 83210

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child272 child173 - child
    bread302 bread53 - bread-portion
    content186 content113 - content-portion
    tray224 tray61 - tray
    table64 table127 table195 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray224 kitchen)
     (at tray61 kitchen)
     (at_kitchen_bread bread302)
     (at_kitchen_bread bread53)
     (at_kitchen_content content186)
     (at_kitchen_content content113)
     (not_allergic_gluten child272)
     (not_allergic_gluten child173)
     (waiting child272 table195)
     (waiting child173 table127)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child272)
     (served child173)
    )
  )
)
