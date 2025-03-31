; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 224182

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child465 child485 - child
    bread343 bread154 - bread-portion
    content52 content54 - content-portion
    tray60 tray472 - tray
    table173 table268 table366 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray60 kitchen)
     (at tray472 kitchen)
     (at_kitchen_bread bread343)
     (at_kitchen_bread bread154)
     (at_kitchen_content content52)
     (at_kitchen_content content54)
     (not_allergic_gluten child465)
     (not_allergic_gluten child485)
     (waiting child465 table268)
     (waiting child485 table173)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child465)
     (served child485)
    )
  )
)
