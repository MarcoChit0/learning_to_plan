; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 359198

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child334 - child
    bread106 - bread-portion
    content111 - content-portion
    tray76 tray90 tray215 - tray
    table166 table301 table3 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray76 kitchen)
     (at tray90 kitchen)
     (at tray215 kitchen)
     (at_kitchen_bread bread106)
     (at_kitchen_content content111)
     (not_allergic_gluten child334)
     (waiting child334 table166)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child334)
    )
  )
)
