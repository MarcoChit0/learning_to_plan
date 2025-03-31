; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 787535

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child466 child243 - child
    bread268 bread450 - bread-portion
    content493 content95 - content-portion
    tray62 tray209 - tray
    table438 table300 table304 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray62 kitchen)
     (at tray209 kitchen)
     (at_kitchen_bread bread268)
     (at_kitchen_bread bread450)
     (at_kitchen_content content493)
     (at_kitchen_content content95)
     (no_gluten_bread bread450)
     (no_gluten_content content95)
     (allergic_gluten child243)
     (not_allergic_gluten child466)
     (waiting child466 table438)
     (waiting child243 table300)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child466)
     (served child243)
    )
  )
)
