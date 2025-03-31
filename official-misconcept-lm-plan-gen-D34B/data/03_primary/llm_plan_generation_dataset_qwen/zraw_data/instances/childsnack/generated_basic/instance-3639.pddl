; child-snack task with 2 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 286411

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child38 child161 - child
    bread263 bread40 - bread-portion
    content492 content499 - content-portion
    tray476 tray133 tray102 tray394 - tray
    table72 table170 table194 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray476 kitchen)
     (at tray133 kitchen)
     (at tray102 kitchen)
     (at tray394 kitchen)
     (at_kitchen_bread bread263)
     (at_kitchen_bread bread40)
     (at_kitchen_content content492)
     (at_kitchen_content content499)
     (no_gluten_bread bread263)
     (no_gluten_content content499)
     (allergic_gluten child161)
     (not_allergic_gluten child38)
     (waiting child38 table194)
     (waiting child161 table170)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child38)
     (served child161)
    )
  )
)
