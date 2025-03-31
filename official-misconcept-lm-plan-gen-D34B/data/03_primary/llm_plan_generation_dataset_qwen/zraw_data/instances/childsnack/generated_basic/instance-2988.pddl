; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 108630

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child291 child309 - child
    bread14 bread357 - bread-portion
    content187 content408 - content-portion
    tray426 tray407 tray58 tray394 - tray
    table229 table268 table146 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray426 kitchen)
     (at tray407 kitchen)
     (at tray58 kitchen)
     (at tray394 kitchen)
     (at_kitchen_bread bread14)
     (at_kitchen_bread bread357)
     (at_kitchen_content content187)
     (at_kitchen_content content408)
     (not_allergic_gluten child309)
     (not_allergic_gluten child291)
     (waiting child291 table146)
     (waiting child309 table229)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child291)
     (served child309)
    )
  )
)
