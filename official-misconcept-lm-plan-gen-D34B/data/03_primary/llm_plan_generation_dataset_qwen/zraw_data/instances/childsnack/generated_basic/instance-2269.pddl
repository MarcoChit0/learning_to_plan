; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 232354

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child212 child391 - child
    bread153 bread275 - bread-portion
    content54 content316 - content-portion
    tray3 - tray
    table170 table394 table381 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray3 kitchen)
     (at_kitchen_bread bread153)
     (at_kitchen_bread bread275)
     (at_kitchen_content content54)
     (at_kitchen_content content316)
     (no_gluten_bread bread153)
     (no_gluten_content content316)
     (allergic_gluten child391)
     (not_allergic_gluten child212)
     (waiting child212 table394)
     (waiting child391 table170)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child212)
     (served child391)
    )
  )
)
