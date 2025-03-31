; child-snack task with 2 children and 0.6 gluten factor 
; constant factor of 1.3
; random seed: 43297

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child73 child393 - child
    bread369 bread267 - bread-portion
    content53 content201 - content-portion
    tray5 tray203 - tray
    table252 table25 table347 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray5 kitchen)
     (at tray203 kitchen)
     (at_kitchen_bread bread369)
     (at_kitchen_bread bread267)
     (at_kitchen_content content53)
     (at_kitchen_content content201)
     (no_gluten_bread bread267)
     (no_gluten_content content53)
     (allergic_gluten child73)
     (not_allergic_gluten child393)
     (waiting child73 table25)
     (waiting child393 table347)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child73)
     (served child393)
    )
  )
)
