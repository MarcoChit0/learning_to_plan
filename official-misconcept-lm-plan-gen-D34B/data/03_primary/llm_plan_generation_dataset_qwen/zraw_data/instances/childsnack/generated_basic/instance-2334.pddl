; child-snack task with 1 children and 0.7 gluten factor 
; constant factor of 1.3
; random seed: 996551

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child451 - child
    bread177 - bread-portion
    content292 - content-portion
    tray403 - tray
    table183 table119 table160 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray403 kitchen)
     (at_kitchen_bread bread177)
     (at_kitchen_content content292)
     (not_allergic_gluten child451)
     (waiting child451 table160)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child451)
    )
  )
)
