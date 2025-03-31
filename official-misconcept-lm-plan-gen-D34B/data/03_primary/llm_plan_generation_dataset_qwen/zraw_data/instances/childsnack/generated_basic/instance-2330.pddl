; child-snack task with 1 children and 0.7 gluten factor 
; constant factor of 1.3
; random seed: 568677

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child230 - child
    bread209 - bread-portion
    content498 - content-portion
    tray477 - tray
    table47 table441 table317 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray477 kitchen)
     (at_kitchen_bread bread209)
     (at_kitchen_content content498)
     (not_allergic_gluten child230)
     (waiting child230 table47)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child230)
    )
  )
)
