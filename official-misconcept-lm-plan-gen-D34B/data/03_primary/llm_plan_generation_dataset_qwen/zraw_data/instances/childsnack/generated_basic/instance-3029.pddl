; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 420041

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child366 child358 - child
    bread399 bread20 - bread-portion
    content134 content243 - content-portion
    tray133 tray488 tray481 tray264 - tray
    table16 table411 table367 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray133 kitchen)
     (at tray488 kitchen)
     (at tray481 kitchen)
     (at tray264 kitchen)
     (at_kitchen_bread bread399)
     (at_kitchen_bread bread20)
     (at_kitchen_content content134)
     (at_kitchen_content content243)
     (not_allergic_gluten child358)
     (not_allergic_gluten child366)
     (waiting child366 table16)
     (waiting child358 table16)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child366)
     (served child358)
    )
  )
)
