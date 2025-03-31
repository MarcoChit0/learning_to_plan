; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 467363

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child4 child61 - child
    bread196 bread121 - bread-portion
    content340 content268 - content-portion
    tray247 - tray
    table327 table140 table244 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray247 kitchen)
     (at_kitchen_bread bread196)
     (at_kitchen_bread bread121)
     (at_kitchen_content content340)
     (at_kitchen_content content268)
     (no_gluten_bread bread121)
     (no_gluten_content content268)
     (allergic_gluten child4)
     (not_allergic_gluten child61)
     (waiting child4 table244)
     (waiting child61 table327)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child4)
     (served child61)
    )
  )
)
