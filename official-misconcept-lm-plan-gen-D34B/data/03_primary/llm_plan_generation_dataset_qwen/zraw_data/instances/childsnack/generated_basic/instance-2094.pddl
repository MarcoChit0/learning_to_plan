; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 95626

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child378 - child
    bread433 - bread-portion
    content103 - content-portion
    tray123 tray407 tray347 tray290 - tray
    table412 table38 table290 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray123 kitchen)
     (at tray407 kitchen)
     (at tray347 kitchen)
     (at tray290 kitchen)
     (at_kitchen_bread bread433)
     (at_kitchen_content content103)
     (not_allergic_gluten child378)
     (waiting child378 table38)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child378)
    )
  )
)
