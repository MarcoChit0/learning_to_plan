; child-snack task with 1 children and 0.63 gluten factor 
; constant factor of 1.5
; random seed: 804168

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child260 - child
    bread334 - bread-portion
    content112 - content-portion
    tray252 - tray
    table404 table386 table34 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray252 kitchen)
     (at_kitchen_bread bread334)
     (at_kitchen_content content112)
     (not_allergic_gluten child260)
     (waiting child260 table404)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child260)
    )
  )
)
