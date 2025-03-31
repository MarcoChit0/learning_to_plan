; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 796365

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child271 child164 - child
    bread10 bread19 - bread-portion
    content125 content79 - content-portion
    tray411 tray475 - tray
    table493 table275 table149 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray411 kitchen)
     (at tray475 kitchen)
     (at_kitchen_bread bread10)
     (at_kitchen_bread bread19)
     (at_kitchen_content content125)
     (at_kitchen_content content79)
     (not_allergic_gluten child164)
     (not_allergic_gluten child271)
     (waiting child271 table149)
     (waiting child164 table149)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child271)
     (served child164)
    )
  )
)
