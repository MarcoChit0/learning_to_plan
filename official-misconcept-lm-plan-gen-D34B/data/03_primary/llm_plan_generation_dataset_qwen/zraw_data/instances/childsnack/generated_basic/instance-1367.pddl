; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 876573

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child81 child150 - child
    bread277 bread263 - bread-portion
    content181 content495 - content-portion
    tray250 tray422 tray403 - tray
    table60 table223 table156 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray250 kitchen)
     (at tray422 kitchen)
     (at tray403 kitchen)
     (at_kitchen_bread bread277)
     (at_kitchen_bread bread263)
     (at_kitchen_content content181)
     (at_kitchen_content content495)
     (not_allergic_gluten child81)
     (not_allergic_gluten child150)
     (waiting child81 table156)
     (waiting child150 table223)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child81)
     (served child150)
    )
  )
)
