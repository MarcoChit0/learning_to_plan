; child-snack task with 2 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 772113

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child85 child439 - child
    bread450 bread42 - bread-portion
    content305 content159 - content-portion
    tray422 tray264 tray272 tray75 - tray
    table211 table185 table213 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray422 kitchen)
     (at tray264 kitchen)
     (at tray272 kitchen)
     (at tray75 kitchen)
     (at_kitchen_bread bread450)
     (at_kitchen_bread bread42)
     (at_kitchen_content content305)
     (at_kitchen_content content159)
     (not_allergic_gluten child85)
     (not_allergic_gluten child439)
     (waiting child85 table211)
     (waiting child439 table213)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child85)
     (served child439)
    )
  )
)
