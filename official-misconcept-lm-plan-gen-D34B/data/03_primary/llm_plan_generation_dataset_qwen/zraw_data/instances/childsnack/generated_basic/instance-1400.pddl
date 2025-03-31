; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 158107

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child450 child235 - child
    bread428 bread265 - bread-portion
    content17 content302 - content-portion
    tray10 - tray
    table160 table142 table243 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray10 kitchen)
     (at_kitchen_bread bread428)
     (at_kitchen_bread bread265)
     (at_kitchen_content content17)
     (at_kitchen_content content302)
     (no_gluten_bread bread428)
     (no_gluten_content content302)
     (allergic_gluten child450)
     (not_allergic_gluten child235)
     (waiting child450 table160)
     (waiting child235 table160)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child450)
     (served child235)
    )
  )
)
