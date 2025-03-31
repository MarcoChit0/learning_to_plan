; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 716903

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child183 - child
    bread33 - bread-portion
    content444 - content-portion
    tray114 - tray
    table198 table352 table344 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray114 kitchen)
     (at_kitchen_bread bread33)
     (at_kitchen_content content444)
     (not_allergic_gluten child183)
     (waiting child183 table352)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child183)
    )
  )
)
