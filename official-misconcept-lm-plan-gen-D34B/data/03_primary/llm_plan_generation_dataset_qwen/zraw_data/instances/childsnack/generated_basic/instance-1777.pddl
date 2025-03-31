; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 450114

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child476 - child
    bread480 - bread-portion
    content324 - content-portion
    tray158 - tray
    table232 table417 table64 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray158 kitchen)
     (at_kitchen_bread bread480)
     (at_kitchen_content content324)
     (not_allergic_gluten child476)
     (waiting child476 table232)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child476)
    )
  )
)
