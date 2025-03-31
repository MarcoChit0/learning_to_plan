; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 256917

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child491 - child
    bread155 - bread-portion
    content40 - content-portion
    tray128 - tray
    table51 table26 table38 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray128 kitchen)
     (at_kitchen_bread bread155)
     (at_kitchen_content content40)
     (not_allergic_gluten child491)
     (waiting child491 table38)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child491)
    )
  )
)
