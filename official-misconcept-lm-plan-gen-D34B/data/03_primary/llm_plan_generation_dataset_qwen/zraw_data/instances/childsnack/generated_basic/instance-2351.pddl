; child-snack task with 1 children and 0.7 gluten factor 
; constant factor of 1.3
; random seed: 581184

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child353 - child
    bread304 - bread-portion
    content302 - content-portion
    tray457 - tray
    table468 table490 table37 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray457 kitchen)
     (at_kitchen_bread bread304)
     (at_kitchen_content content302)
     (not_allergic_gluten child353)
     (waiting child353 table37)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child353)
    )
  )
)
