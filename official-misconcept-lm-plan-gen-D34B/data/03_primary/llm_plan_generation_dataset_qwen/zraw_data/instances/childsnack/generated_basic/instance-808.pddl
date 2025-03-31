; child-snack task with 1 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 756657

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child322 - child
    bread28 - bread-portion
    content179 - content-portion
    tray383 tray249 - tray
    table115 table473 table383 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray383 kitchen)
     (at tray249 kitchen)
     (at_kitchen_bread bread28)
     (at_kitchen_content content179)
     (not_allergic_gluten child322)
     (waiting child322 table115)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child322)
    )
  )
)
