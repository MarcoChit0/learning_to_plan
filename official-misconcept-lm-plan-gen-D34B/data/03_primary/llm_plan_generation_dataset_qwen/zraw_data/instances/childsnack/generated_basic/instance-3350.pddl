; child-snack task with 2 children and 0.6 gluten factor 
; constant factor of 1.3
; random seed: 469598

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child433 child13 - child
    bread115 bread126 - bread-portion
    content210 content29 - content-portion
    tray192 tray223 - tray
    table300 table148 table226 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray192 kitchen)
     (at tray223 kitchen)
     (at_kitchen_bread bread115)
     (at_kitchen_bread bread126)
     (at_kitchen_content content210)
     (at_kitchen_content content29)
     (no_gluten_bread bread115)
     (no_gluten_content content29)
     (allergic_gluten child13)
     (not_allergic_gluten child433)
     (waiting child433 table300)
     (waiting child13 table148)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child433)
     (served child13)
    )
  )
)
