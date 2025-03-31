; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 858078

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child488 child73 - child
    bread254 bread35 - bread-portion
    content499 content427 - content-portion
    tray476 - tray
    table437 table268 table351 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray476 kitchen)
     (at_kitchen_bread bread254)
     (at_kitchen_bread bread35)
     (at_kitchen_content content499)
     (at_kitchen_content content427)
     (no_gluten_bread bread254)
     (no_gluten_content content499)
     (allergic_gluten child73)
     (not_allergic_gluten child488)
     (waiting child488 table351)
     (waiting child73 table437)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child488)
     (served child73)
    )
  )
)
