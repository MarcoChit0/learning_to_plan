; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 291092

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child199 child358 - child
    bread130 bread453 - bread-portion
    content263 content59 - content-portion
    tray357 tray9 tray49 tray245 - tray
    table47 table35 table380 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray357 kitchen)
     (at tray9 kitchen)
     (at tray49 kitchen)
     (at tray245 kitchen)
     (at_kitchen_bread bread130)
     (at_kitchen_bread bread453)
     (at_kitchen_content content263)
     (at_kitchen_content content59)
     (not_allergic_gluten child199)
     (not_allergic_gluten child358)
     (waiting child199 table35)
     (waiting child358 table47)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child199)
     (served child358)
    )
  )
)
