; child-snack task with 2 children and 0.66 gluten factor 
; constant factor of 1.3
; random seed: 280964

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child109 child322 - child
    bread286 bread285 - bread-portion
    content134 content312 - content-portion
    tray483 tray382 tray145 - tray
    table426 table231 table105 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray483 kitchen)
     (at tray382 kitchen)
     (at tray145 kitchen)
     (at_kitchen_bread bread286)
     (at_kitchen_bread bread285)
     (at_kitchen_content content134)
     (at_kitchen_content content312)
     (no_gluten_bread bread286)
     (no_gluten_content content312)
     (allergic_gluten child322)
     (not_allergic_gluten child109)
     (waiting child109 table426)
     (waiting child322 table105)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child109)
     (served child322)
    )
  )
)
