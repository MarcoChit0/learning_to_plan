; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 259627

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child456 - child
    bread164 - bread-portion
    content182 - content-portion
    tray176 tray341 - tray
    table445 table463 table140 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray176 kitchen)
     (at tray341 kitchen)
     (at_kitchen_bread bread164)
     (at_kitchen_content content182)
     (not_allergic_gluten child456)
     (waiting child456 table140)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child456)
    )
  )
)
