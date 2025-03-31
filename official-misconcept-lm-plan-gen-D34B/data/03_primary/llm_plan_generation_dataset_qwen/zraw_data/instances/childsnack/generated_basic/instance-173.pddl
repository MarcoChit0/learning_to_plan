; child-snack task with 2 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 632797

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child98 child462 - child
    bread285 bread294 - bread-portion
    content216 content474 - content-portion
    tray35 tray148 tray142 tray132 - tray
    table386 table69 table260 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray35 kitchen)
     (at tray148 kitchen)
     (at tray142 kitchen)
     (at tray132 kitchen)
     (at_kitchen_bread bread285)
     (at_kitchen_bread bread294)
     (at_kitchen_content content216)
     (at_kitchen_content content474)
     (not_allergic_gluten child462)
     (not_allergic_gluten child98)
     (waiting child98 table386)
     (waiting child462 table386)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child98)
     (served child462)
    )
  )
)
