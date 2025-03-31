; child-snack task with 1 children and 0.7 gluten factor 
; constant factor of 1.3
; random seed: 760395

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child345 - child
    bread381 - bread-portion
    content70 - content-portion
    tray329 - tray
    table252 table177 table83 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray329 kitchen)
     (at_kitchen_bread bread381)
     (at_kitchen_content content70)
     (not_allergic_gluten child345)
     (waiting child345 table252)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child345)
    )
  )
)
