; child-snack task with 2 children and 0.53 gluten factor 
; constant factor of 1.9
; random seed: 794082

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child29 child120 - child
    bread187 bread158 - bread-portion
    content311 content359 - content-portion
    tray24 tray78 tray462 - tray
    table211 table480 table267 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray24 kitchen)
     (at tray78 kitchen)
     (at tray462 kitchen)
     (at_kitchen_bread bread187)
     (at_kitchen_bread bread158)
     (at_kitchen_content content311)
     (at_kitchen_content content359)
     (no_gluten_bread bread187)
     (no_gluten_content content359)
     (allergic_gluten child120)
     (not_allergic_gluten child29)
     (waiting child29 table267)
     (waiting child120 table211)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child29)
     (served child120)
    )
  )
)
