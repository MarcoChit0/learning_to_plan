; child-snack task with 2 children and 0.59 gluten factor 
; constant factor of 1.9
; random seed: 241501

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child120 child211 - child
    bread443 bread469 - bread-portion
    content69 content110 - content-portion
    tray264 tray363 - tray
    table221 table315 table164 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray264 kitchen)
     (at tray363 kitchen)
     (at_kitchen_bread bread443)
     (at_kitchen_bread bread469)
     (at_kitchen_content content69)
     (at_kitchen_content content110)
     (no_gluten_bread bread443)
     (no_gluten_content content110)
     (allergic_gluten child211)
     (not_allergic_gluten child120)
     (waiting child120 table315)
     (waiting child211 table221)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child120)
     (served child211)
    )
  )
)
