; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 936448

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child205 child330 - child
    bread261 bread83 - bread-portion
    content52 content428 - content-portion
    tray18 - tray
    table230 table304 table185 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray18 kitchen)
     (at_kitchen_bread bread261)
     (at_kitchen_bread bread83)
     (at_kitchen_content content52)
     (at_kitchen_content content428)
     (no_gluten_bread bread261)
     (no_gluten_content content52)
     (allergic_gluten child330)
     (not_allergic_gluten child205)
     (waiting child205 table185)
     (waiting child330 table230)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child205)
     (served child330)
    )
  )
)
