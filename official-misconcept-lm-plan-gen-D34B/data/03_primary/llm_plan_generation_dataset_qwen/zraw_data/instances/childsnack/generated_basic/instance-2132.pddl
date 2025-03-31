; child-snack task with 2 children and 0.53 gluten factor 
; constant factor of 1.9
; random seed: 336247

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child309 child323 - child
    bread221 bread209 - bread-portion
    content325 content482 - content-portion
    tray189 tray94 tray306 - tray
    table22 table136 table279 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray189 kitchen)
     (at tray94 kitchen)
     (at tray306 kitchen)
     (at_kitchen_bread bread221)
     (at_kitchen_bread bread209)
     (at_kitchen_content content325)
     (at_kitchen_content content482)
     (no_gluten_bread bread221)
     (no_gluten_content content482)
     (allergic_gluten child309)
     (not_allergic_gluten child323)
     (waiting child309 table279)
     (waiting child323 table22)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child309)
     (served child323)
    )
  )
)
