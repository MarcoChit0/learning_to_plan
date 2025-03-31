; child-snack task with 1 children and 0.69 gluten factor 
; constant factor of 1.3
; random seed: 771992

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child129 - child
    bread212 - bread-portion
    content340 - content-portion
    tray331 tray109 tray193 tray360 - tray
    table478 table349 table155 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray331 kitchen)
     (at tray109 kitchen)
     (at tray193 kitchen)
     (at tray360 kitchen)
     (at_kitchen_bread bread212)
     (at_kitchen_content content340)
     (not_allergic_gluten child129)
     (waiting child129 table478)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child129)
    )
  )
)
