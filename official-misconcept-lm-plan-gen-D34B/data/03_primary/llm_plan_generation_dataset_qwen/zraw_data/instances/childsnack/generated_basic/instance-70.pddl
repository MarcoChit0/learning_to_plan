; child-snack task with 2 children and 0.36 gluten factor 
; constant factor of 1.9
; random seed: 765126

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child437 child288 - child
    bread82 bread311 - bread-portion
    content453 content12 - content-portion
    tray8 tray37 - tray
    table21 table251 table232 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray8 kitchen)
     (at tray37 kitchen)
     (at_kitchen_bread bread82)
     (at_kitchen_bread bread311)
     (at_kitchen_content content453)
     (at_kitchen_content content12)
     (not_allergic_gluten child288)
     (not_allergic_gluten child437)
     (waiting child437 table232)
     (waiting child288 table251)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child437)
     (served child288)
    )
  )
)
