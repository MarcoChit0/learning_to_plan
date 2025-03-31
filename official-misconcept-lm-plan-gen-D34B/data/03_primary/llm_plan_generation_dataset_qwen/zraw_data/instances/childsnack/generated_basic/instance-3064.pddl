; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 760848

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child432 child195 - child
    bread314 bread306 - bread-portion
    content327 content304 - content-portion
    tray227 tray95 tray312 - tray
    table12 table386 table121 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray227 kitchen)
     (at tray95 kitchen)
     (at tray312 kitchen)
     (at_kitchen_bread bread314)
     (at_kitchen_bread bread306)
     (at_kitchen_content content327)
     (at_kitchen_content content304)
     (no_gluten_bread bread306)
     (no_gluten_content content304)
     (allergic_gluten child195)
     (not_allergic_gluten child432)
     (waiting child432 table12)
     (waiting child195 table121)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child432)
     (served child195)
    )
  )
)
