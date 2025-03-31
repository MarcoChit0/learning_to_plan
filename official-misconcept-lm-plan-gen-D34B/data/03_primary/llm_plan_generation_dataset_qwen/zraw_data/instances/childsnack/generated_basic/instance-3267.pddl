; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 716134

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child348 child147 - child
    bread218 bread143 - bread-portion
    content78 content273 - content-portion
    tray307 tray418 tray248 tray498 - tray
    table485 table118 table15 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray307 kitchen)
     (at tray418 kitchen)
     (at tray248 kitchen)
     (at tray498 kitchen)
     (at_kitchen_bread bread218)
     (at_kitchen_bread bread143)
     (at_kitchen_content content78)
     (at_kitchen_content content273)
     (no_gluten_bread bread143)
     (no_gluten_content content273)
     (allergic_gluten child348)
     (not_allergic_gluten child147)
     (waiting child348 table485)
     (waiting child147 table15)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child348)
     (served child147)
    )
  )
)
