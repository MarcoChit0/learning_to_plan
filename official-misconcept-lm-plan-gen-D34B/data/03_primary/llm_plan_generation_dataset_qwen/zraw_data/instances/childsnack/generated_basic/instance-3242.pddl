; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.9
; random seed: 813095

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child271 child172 - child
    bread259 bread428 - bread-portion
    content69 content236 - content-portion
    tray352 tray473 tray168 tray424 - tray
    table299 table41 table56 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray352 kitchen)
     (at tray473 kitchen)
     (at tray168 kitchen)
     (at tray424 kitchen)
     (at_kitchen_bread bread259)
     (at_kitchen_bread bread428)
     (at_kitchen_content content69)
     (at_kitchen_content content236)
     (no_gluten_bread bread428)
     (no_gluten_content content236)
     (allergic_gluten child271)
     (not_allergic_gluten child172)
     (waiting child271 table299)
     (waiting child172 table56)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child271)
     (served child172)
    )
  )
)
