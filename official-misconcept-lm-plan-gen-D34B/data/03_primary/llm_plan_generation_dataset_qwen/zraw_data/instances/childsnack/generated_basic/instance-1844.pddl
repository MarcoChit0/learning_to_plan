; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.9
; random seed: 147537

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child367 child41 - child
    bread294 bread188 - bread-portion
    content209 content404 - content-portion
    tray27 - tray
    table125 table315 table161 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray27 kitchen)
     (at_kitchen_bread bread294)
     (at_kitchen_bread bread188)
     (at_kitchen_content content209)
     (at_kitchen_content content404)
     (no_gluten_bread bread188)
     (no_gluten_content content209)
     (allergic_gluten child367)
     (not_allergic_gluten child41)
     (waiting child367 table315)
     (waiting child41 table161)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child367)
     (served child41)
    )
  )
)
