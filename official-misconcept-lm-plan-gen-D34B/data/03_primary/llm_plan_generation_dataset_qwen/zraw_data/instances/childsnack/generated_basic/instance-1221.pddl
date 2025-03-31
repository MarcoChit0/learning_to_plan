; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 635018

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child358 child377 child282 - child
    bread43 bread442 bread450 - bread-portion
    content93 content385 content206 - content-portion
    tray293 tray333 tray313 - tray
    table154 table318 table394 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray293 kitchen)
     (at tray333 kitchen)
     (at tray313 kitchen)
     (at_kitchen_bread bread43)
     (at_kitchen_bread bread442)
     (at_kitchen_bread bread450)
     (at_kitchen_content content93)
     (at_kitchen_content content385)
     (at_kitchen_content content206)
     (not_allergic_gluten child282)
     (not_allergic_gluten child358)
     (not_allergic_gluten child377)
     (waiting child358 table154)
     (waiting child377 table394)
     (waiting child282 table394)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child358)
     (served child377)
     (served child282)
    )
  )
)
