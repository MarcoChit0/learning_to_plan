; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 501031

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child167 - child
    bread114 - bread-portion
    content388 - content-portion
    tray248 tray394 - tray
    table402 table359 table329 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray248 kitchen)
     (at tray394 kitchen)
     (at_kitchen_bread bread114)
     (at_kitchen_content content388)
     (not_allergic_gluten child167)
     (waiting child167 table359)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child167)
    )
  )
)
