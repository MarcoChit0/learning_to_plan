; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 343476

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child42 child93 - child
    bread313 bread402 - bread-portion
    content325 content182 - content-portion
    tray341 tray86 - tray
    table203 table105 table330 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray341 kitchen)
     (at tray86 kitchen)
     (at_kitchen_bread bread313)
     (at_kitchen_bread bread402)
     (at_kitchen_content content325)
     (at_kitchen_content content182)
     (no_gluten_bread bread402)
     (no_gluten_content content325)
     (allergic_gluten child93)
     (not_allergic_gluten child42)
     (waiting child42 table105)
     (waiting child93 table203)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child42)
     (served child93)
    )
  )
)
