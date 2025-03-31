; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 558957

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child53 child292 - child
    bread139 bread75 - bread-portion
    content283 content443 - content-portion
    tray69 tray323 tray53 tray369 - tray
    table355 table96 table419 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray69 kitchen)
     (at tray323 kitchen)
     (at tray53 kitchen)
     (at tray369 kitchen)
     (at_kitchen_bread bread139)
     (at_kitchen_bread bread75)
     (at_kitchen_content content283)
     (at_kitchen_content content443)
     (not_allergic_gluten child292)
     (not_allergic_gluten child53)
     (waiting child53 table419)
     (waiting child292 table96)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child53)
     (served child292)
    )
  )
)
