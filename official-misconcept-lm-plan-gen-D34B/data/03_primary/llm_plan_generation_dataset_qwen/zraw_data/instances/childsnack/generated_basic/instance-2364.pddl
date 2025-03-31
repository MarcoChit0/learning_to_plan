; child-snack task with 2 children and 0.38 gluten factor 
; constant factor of 1.9
; random seed: 548700

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child358 child374 - child
    bread182 bread386 - bread-portion
    content199 content326 - content-portion
    tray230 - tray
    table371 table73 table398 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray230 kitchen)
     (at_kitchen_bread bread182)
     (at_kitchen_bread bread386)
     (at_kitchen_content content199)
     (at_kitchen_content content326)
     (not_allergic_gluten child358)
     (not_allergic_gluten child374)
     (waiting child358 table371)
     (waiting child374 table371)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child358)
     (served child374)
    )
  )
)
