; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 822891

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child313 child161 - child
    bread349 bread387 - bread-portion
    content163 content445 - content-portion
    tray128 - tray
    table180 table315 table490 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray128 kitchen)
     (at_kitchen_bread bread349)
     (at_kitchen_bread bread387)
     (at_kitchen_content content163)
     (at_kitchen_content content445)
     (no_gluten_bread bread349)
     (no_gluten_content content445)
     (allergic_gluten child161)
     (not_allergic_gluten child313)
     (waiting child313 table315)
     (waiting child161 table490)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child313)
     (served child161)
    )
  )
)
