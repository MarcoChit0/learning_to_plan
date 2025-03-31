; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 310401

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child422 child364 - child
    bread238 bread320 - bread-portion
    content305 content182 - content-portion
    tray301 tray55 tray354 - tray
    table426 table469 table455 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray301 kitchen)
     (at tray55 kitchen)
     (at tray354 kitchen)
     (at_kitchen_bread bread238)
     (at_kitchen_bread bread320)
     (at_kitchen_content content305)
     (at_kitchen_content content182)
     (not_allergic_gluten child364)
     (not_allergic_gluten child422)
     (waiting child422 table455)
     (waiting child364 table426)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child422)
     (served child364)
    )
  )
)
