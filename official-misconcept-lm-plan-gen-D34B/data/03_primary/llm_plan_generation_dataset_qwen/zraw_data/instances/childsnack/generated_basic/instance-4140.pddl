; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 531209

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child333 child490 - child
    bread291 bread306 - bread-portion
    content180 content333 - content-portion
    tray330 tray120 tray204 - tray
    table165 table14 table462 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray330 kitchen)
     (at tray120 kitchen)
     (at tray204 kitchen)
     (at_kitchen_bread bread291)
     (at_kitchen_bread bread306)
     (at_kitchen_content content180)
     (at_kitchen_content content333)
     (not_allergic_gluten child333)
     (not_allergic_gluten child490)
     (waiting child333 table165)
     (waiting child490 table14)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child333)
     (served child490)
    )
  )
)
