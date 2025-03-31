; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 6786

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child452 child377 - child
    bread374 bread166 - bread-portion
    content24 content30 - content-portion
    tray388 tray177 - tray
    table96 table140 table139 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray388 kitchen)
     (at tray177 kitchen)
     (at_kitchen_bread bread374)
     (at_kitchen_bread bread166)
     (at_kitchen_content content24)
     (at_kitchen_content content30)
     (not_allergic_gluten child377)
     (not_allergic_gluten child452)
     (waiting child452 table139)
     (waiting child377 table139)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child452)
     (served child377)
    )
  )
)
