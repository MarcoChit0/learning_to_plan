; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 850884

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child108 - child
    bread481 - bread-portion
    content3 - content-portion
    tray498 tray419 tray426 tray433 - tray
    table178 table418 table458 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray498 kitchen)
     (at tray419 kitchen)
     (at tray426 kitchen)
     (at tray433 kitchen)
     (at_kitchen_bread bread481)
     (at_kitchen_content content3)
     (not_allergic_gluten child108)
     (waiting child108 table458)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child108)
    )
  )
)
