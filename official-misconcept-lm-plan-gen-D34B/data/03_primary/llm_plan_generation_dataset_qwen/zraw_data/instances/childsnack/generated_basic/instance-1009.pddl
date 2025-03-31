; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 689797

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child343 child283 child360 - child
    bread349 bread212 bread497 - bread-portion
    content97 content22 content202 - content-portion
    tray351 - tray
    table20 table131 table279 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray351 kitchen)
     (at_kitchen_bread bread349)
     (at_kitchen_bread bread212)
     (at_kitchen_bread bread497)
     (at_kitchen_content content97)
     (at_kitchen_content content22)
     (at_kitchen_content content202)
     (not_allergic_gluten child360)
     (not_allergic_gluten child283)
     (not_allergic_gluten child343)
     (waiting child343 table279)
     (waiting child283 table279)
     (waiting child360 table20)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child343)
     (served child283)
     (served child360)
    )
  )
)
