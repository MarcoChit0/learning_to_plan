; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 808967

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child184 - child
    bread311 - bread-portion
    content439 - content-portion
    tray447 tray151 tray303 - tray
    table244 table204 table417 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray447 kitchen)
     (at tray151 kitchen)
     (at tray303 kitchen)
     (at_kitchen_bread bread311)
     (at_kitchen_content content439)
     (not_allergic_gluten child184)
     (waiting child184 table204)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child184)
    )
  )
)
