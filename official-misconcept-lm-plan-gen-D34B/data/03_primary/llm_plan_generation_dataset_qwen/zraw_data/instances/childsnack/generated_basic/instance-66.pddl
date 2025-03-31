; child-snack task with 2 children and 0.36 gluten factor 
; constant factor of 1.9
; random seed: 143164

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child162 child360 - child
    bread281 bread421 - bread-portion
    content274 content154 - content-portion
    tray404 tray467 - tray
    table461 table170 table302 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray404 kitchen)
     (at tray467 kitchen)
     (at_kitchen_bread bread281)
     (at_kitchen_bread bread421)
     (at_kitchen_content content274)
     (at_kitchen_content content154)
     (not_allergic_gluten child360)
     (not_allergic_gluten child162)
     (waiting child162 table302)
     (waiting child360 table461)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child162)
     (served child360)
    )
  )
)
