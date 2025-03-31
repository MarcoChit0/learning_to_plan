; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.3
; random seed: 624386

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child151 child319 - child
    bread109 bread24 - bread-portion
    content169 content482 - content-portion
    tray390 - tray
    table139 table136 table187 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray390 kitchen)
     (at_kitchen_bread bread109)
     (at_kitchen_bread bread24)
     (at_kitchen_content content169)
     (at_kitchen_content content482)
     (no_gluten_bread bread109)
     (no_gluten_content content482)
     (allergic_gluten child319)
     (not_allergic_gluten child151)
     (waiting child151 table136)
     (waiting child319 table139)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child151)
     (served child319)
    )
  )
)
