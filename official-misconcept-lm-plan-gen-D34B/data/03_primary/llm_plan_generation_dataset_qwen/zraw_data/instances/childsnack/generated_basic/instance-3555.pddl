; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 975916

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child258 child257 - child
    bread396 bread88 - bread-portion
    content182 content495 - content-portion
    tray270 tray28 - tray
    table310 table462 table222 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray270 kitchen)
     (at tray28 kitchen)
     (at_kitchen_bread bread396)
     (at_kitchen_bread bread88)
     (at_kitchen_content content182)
     (at_kitchen_content content495)
     (not_allergic_gluten child257)
     (not_allergic_gluten child258)
     (waiting child258 table222)
     (waiting child257 table222)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child258)
     (served child257)
    )
  )
)
