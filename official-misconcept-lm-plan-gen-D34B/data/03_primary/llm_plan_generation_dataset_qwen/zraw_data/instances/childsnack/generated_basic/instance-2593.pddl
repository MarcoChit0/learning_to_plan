; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 577422

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child326 - child
    bread472 - bread-portion
    content376 - content-portion
    tray103 tray34 tray389 - tray
    table275 table182 table453 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray103 kitchen)
     (at tray34 kitchen)
     (at tray389 kitchen)
     (at_kitchen_bread bread472)
     (at_kitchen_content content376)
     (not_allergic_gluten child326)
     (waiting child326 table275)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child326)
    )
  )
)
