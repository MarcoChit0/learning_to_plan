; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 155685

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child352 child117 - child
    bread367 bread307 - bread-portion
    content316 content183 - content-portion
    tray365 tray419 tray36 - tray
    table161 table132 table14 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray365 kitchen)
     (at tray419 kitchen)
     (at tray36 kitchen)
     (at_kitchen_bread bread367)
     (at_kitchen_bread bread307)
     (at_kitchen_content content316)
     (at_kitchen_content content183)
     (not_allergic_gluten child117)
     (not_allergic_gluten child352)
     (waiting child352 table14)
     (waiting child117 table14)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child352)
     (served child117)
    )
  )
)
