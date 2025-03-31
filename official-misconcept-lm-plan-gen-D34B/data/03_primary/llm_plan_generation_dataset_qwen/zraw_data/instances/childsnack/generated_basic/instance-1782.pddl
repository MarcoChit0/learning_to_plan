; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 843152

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child229 - child
    bread440 - bread-portion
    content36 - content-portion
    tray397 - tray
    table211 table492 table494 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray397 kitchen)
     (at_kitchen_bread bread440)
     (at_kitchen_content content36)
     (not_allergic_gluten child229)
     (waiting child229 table211)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child229)
    )
  )
)
