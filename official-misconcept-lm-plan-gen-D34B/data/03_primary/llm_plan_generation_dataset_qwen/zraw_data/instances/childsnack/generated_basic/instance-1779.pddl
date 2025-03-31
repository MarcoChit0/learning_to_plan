; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 902913

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child252 - child
    bread453 - bread-portion
    content353 - content-portion
    tray15 - tray
    table30 table204 table266 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray15 kitchen)
     (at_kitchen_bread bread453)
     (at_kitchen_content content353)
     (not_allergic_gluten child252)
     (waiting child252 table30)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child252)
    )
  )
)
