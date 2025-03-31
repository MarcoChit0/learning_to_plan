; child-snack task with 1 children and 0.7 gluten factor 
; constant factor of 1.3
; random seed: 102641

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child424 - child
    bread473 - bread-portion
    content469 - content-portion
    tray210 - tray
    table97 table362 table42 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray210 kitchen)
     (at_kitchen_bread bread473)
     (at_kitchen_content content469)
     (not_allergic_gluten child424)
     (waiting child424 table362)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child424)
    )
  )
)
