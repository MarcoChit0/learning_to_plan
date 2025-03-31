; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 330511

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child334 child408 - child
    bread178 bread24 - bread-portion
    content75 content387 - content-portion
    tray229 - tray
    table451 table309 table420 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray229 kitchen)
     (at_kitchen_bread bread178)
     (at_kitchen_bread bread24)
     (at_kitchen_content content75)
     (at_kitchen_content content387)
     (no_gluten_bread bread24)
     (no_gluten_content content75)
     (allergic_gluten child334)
     (not_allergic_gluten child408)
     (waiting child334 table309)
     (waiting child408 table309)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child334)
     (served child408)
    )
  )
)
