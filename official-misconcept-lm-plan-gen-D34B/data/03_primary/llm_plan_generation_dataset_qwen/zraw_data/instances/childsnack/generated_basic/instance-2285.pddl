; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 76313

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child263 child383 - child
    bread27 bread339 - bread-portion
    content451 content302 - content-portion
    tray380 - tray
    table5 table168 table303 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray380 kitchen)
     (at_kitchen_bread bread27)
     (at_kitchen_bread bread339)
     (at_kitchen_content content451)
     (at_kitchen_content content302)
     (no_gluten_bread bread27)
     (no_gluten_content content451)
     (allergic_gluten child263)
     (not_allergic_gluten child383)
     (waiting child263 table5)
     (waiting child383 table303)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child263)
     (served child383)
    )
  )
)
