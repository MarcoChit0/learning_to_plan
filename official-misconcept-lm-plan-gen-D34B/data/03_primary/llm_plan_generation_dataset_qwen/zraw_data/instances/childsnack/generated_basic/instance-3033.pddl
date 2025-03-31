; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 605785

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child156 child99 - child
    bread254 bread181 - bread-portion
    content286 content247 - content-portion
    tray164 tray57 tray480 - tray
    table66 table404 table101 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray164 kitchen)
     (at tray57 kitchen)
     (at tray480 kitchen)
     (at_kitchen_bread bread254)
     (at_kitchen_bread bread181)
     (at_kitchen_content content286)
     (at_kitchen_content content247)
     (no_gluten_bread bread181)
     (no_gluten_content content247)
     (allergic_gluten child99)
     (not_allergic_gluten child156)
     (waiting child156 table101)
     (waiting child99 table404)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child156)
     (served child99)
    )
  )
)
