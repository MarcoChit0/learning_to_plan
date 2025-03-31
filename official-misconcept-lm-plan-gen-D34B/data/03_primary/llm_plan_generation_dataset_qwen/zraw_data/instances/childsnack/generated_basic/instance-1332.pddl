; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 971889

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child49 child340 - child
    bread84 bread279 - bread-portion
    content127 content371 - content-portion
    tray234 tray102 tray176 - tray
    table243 table117 table274 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray234 kitchen)
     (at tray102 kitchen)
     (at tray176 kitchen)
     (at_kitchen_bread bread84)
     (at_kitchen_bread bread279)
     (at_kitchen_content content127)
     (at_kitchen_content content371)
     (not_allergic_gluten child340)
     (not_allergic_gluten child49)
     (waiting child49 table117)
     (waiting child340 table117)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child49)
     (served child340)
    )
  )
)
