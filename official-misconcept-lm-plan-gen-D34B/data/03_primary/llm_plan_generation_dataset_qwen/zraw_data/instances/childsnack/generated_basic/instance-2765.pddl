; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 295342

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child250 child147 - child
    bread46 bread89 - bread-portion
    content250 content270 - content-portion
    tray231 - tray
    table21 table98 table107 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray231 kitchen)
     (at_kitchen_bread bread46)
     (at_kitchen_bread bread89)
     (at_kitchen_content content250)
     (at_kitchen_content content270)
     (no_gluten_bread bread89)
     (no_gluten_content content270)
     (allergic_gluten child250)
     (not_allergic_gluten child147)
     (waiting child250 table21)
     (waiting child147 table107)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child250)
     (served child147)
    )
  )
)
