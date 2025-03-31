; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 188194

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child473 child126 - child
    bread78 bread35 - bread-portion
    content493 content291 - content-portion
    tray138 tray91 tray348 tray70 - tray
    table209 table9 table192 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray138 kitchen)
     (at tray91 kitchen)
     (at tray348 kitchen)
     (at tray70 kitchen)
     (at_kitchen_bread bread78)
     (at_kitchen_bread bread35)
     (at_kitchen_content content493)
     (at_kitchen_content content291)
     (no_gluten_bread bread35)
     (no_gluten_content content493)
     (allergic_gluten child473)
     (not_allergic_gluten child126)
     (waiting child473 table209)
     (waiting child126 table192)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child473)
     (served child126)
    )
  )
)
