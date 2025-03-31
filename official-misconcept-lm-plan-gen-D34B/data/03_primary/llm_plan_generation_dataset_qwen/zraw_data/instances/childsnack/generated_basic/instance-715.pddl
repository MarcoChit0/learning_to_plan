; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 272877

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child354 - child
    bread71 - bread-portion
    content282 - content-portion
    tray109 tray461 tray175 tray44 - tray
    table5 table188 table207 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray109 kitchen)
     (at tray461 kitchen)
     (at tray175 kitchen)
     (at tray44 kitchen)
     (at_kitchen_bread bread71)
     (at_kitchen_content content282)
     (not_allergic_gluten child354)
     (waiting child354 table5)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child354)
    )
  )
)
