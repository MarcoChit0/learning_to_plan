; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 222879

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child241 - child
    bread198 - bread-portion
    content157 - content-portion
    tray124 tray494 tray340 - tray
    table489 table88 table116 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray124 kitchen)
     (at tray494 kitchen)
     (at tray340 kitchen)
     (at_kitchen_bread bread198)
     (at_kitchen_content content157)
     (not_allergic_gluten child241)
     (waiting child241 table88)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child241)
    )
  )
)
