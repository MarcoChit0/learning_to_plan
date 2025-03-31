; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.7
; random seed: 110408

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child204 child303 - child
    bread69 bread325 - bread-portion
    content177 content203 - content-portion
    tray19 tray423 - tray
    table205 table256 table334 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray19 kitchen)
     (at tray423 kitchen)
     (at_kitchen_bread bread69)
     (at_kitchen_bread bread325)
     (at_kitchen_content content177)
     (at_kitchen_content content203)
     (no_gluten_bread bread325)
     (no_gluten_content content203)
     (allergic_gluten child204)
     (not_allergic_gluten child303)
     (waiting child204 table205)
     (waiting child303 table256)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child204)
     (served child303)
    )
  )
)
