; child-snack task with 2 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 536168

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child324 child87 - child
    bread154 bread348 - bread-portion
    content348 content374 - content-portion
    tray136 tray195 tray376 tray234 - tray
    table140 table177 table324 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray136 kitchen)
     (at tray195 kitchen)
     (at tray376 kitchen)
     (at tray234 kitchen)
     (at_kitchen_bread bread154)
     (at_kitchen_bread bread348)
     (at_kitchen_content content348)
     (at_kitchen_content content374)
     (no_gluten_bread bread348)
     (no_gluten_content content374)
     (allergic_gluten child324)
     (not_allergic_gluten child87)
     (waiting child324 table177)
     (waiting child87 table324)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child324)
     (served child87)
    )
  )
)
