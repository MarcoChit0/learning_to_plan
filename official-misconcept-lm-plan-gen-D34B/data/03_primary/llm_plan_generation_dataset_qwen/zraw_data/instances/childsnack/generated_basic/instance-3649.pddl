; child-snack task with 2 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 444013

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child246 child159 - child
    bread252 bread38 - bread-portion
    content212 content292 - content-portion
    tray98 tray360 tray164 tray404 - tray
    table468 table152 table29 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray98 kitchen)
     (at tray360 kitchen)
     (at tray164 kitchen)
     (at tray404 kitchen)
     (at_kitchen_bread bread252)
     (at_kitchen_bread bread38)
     (at_kitchen_content content212)
     (at_kitchen_content content292)
     (no_gluten_bread bread38)
     (no_gluten_content content292)
     (allergic_gluten child159)
     (not_allergic_gluten child246)
     (waiting child246 table152)
     (waiting child159 table29)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child246)
     (served child159)
    )
  )
)
