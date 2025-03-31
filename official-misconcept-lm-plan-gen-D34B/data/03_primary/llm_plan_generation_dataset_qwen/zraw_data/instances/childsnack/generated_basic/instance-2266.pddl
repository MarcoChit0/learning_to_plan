; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 9186

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child245 child157 - child
    bread88 bread439 - bread-portion
    content77 content370 - content-portion
    tray218 - tray
    table219 table48 table438 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray218 kitchen)
     (at_kitchen_bread bread88)
     (at_kitchen_bread bread439)
     (at_kitchen_content content77)
     (at_kitchen_content content370)
     (no_gluten_bread bread88)
     (no_gluten_content content370)
     (allergic_gluten child157)
     (not_allergic_gluten child245)
     (waiting child245 table438)
     (waiting child157 table438)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child245)
     (served child157)
    )
  )
)
