; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 302252

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child137 child383 - child
    bread241 bread351 - bread-portion
    content382 content177 - content-portion
    tray301 tray42 - tray
    table125 table483 table441 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray301 kitchen)
     (at tray42 kitchen)
     (at_kitchen_bread bread241)
     (at_kitchen_bread bread351)
     (at_kitchen_content content382)
     (at_kitchen_content content177)
     (not_allergic_gluten child137)
     (not_allergic_gluten child383)
     (waiting child137 table441)
     (waiting child383 table125)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child137)
     (served child383)
    )
  )
)
