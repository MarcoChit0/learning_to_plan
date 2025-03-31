; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.7
; random seed: 942741

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child451 child143 - child
    bread332 bread126 - bread-portion
    content24 content151 - content-portion
    tray124 tray338 - tray
    table270 table215 table315 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray124 kitchen)
     (at tray338 kitchen)
     (at_kitchen_bread bread332)
     (at_kitchen_bread bread126)
     (at_kitchen_content content24)
     (at_kitchen_content content151)
     (no_gluten_bread bread332)
     (no_gluten_content content24)
     (allergic_gluten child451)
     (not_allergic_gluten child143)
     (waiting child451 table315)
     (waiting child143 table215)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child451)
     (served child143)
    )
  )
)
