; child-snack task with 2 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 611070

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child108 child298 - child
    bread316 bread418 - bread-portion
    content246 content408 - content-portion
    tray428 tray482 tray203 tray78 - tray
    table220 table25 table303 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray428 kitchen)
     (at tray482 kitchen)
     (at tray203 kitchen)
     (at tray78 kitchen)
     (at_kitchen_bread bread316)
     (at_kitchen_bread bread418)
     (at_kitchen_content content246)
     (at_kitchen_content content408)
     (not_allergic_gluten child108)
     (not_allergic_gluten child298)
     (waiting child108 table303)
     (waiting child298 table303)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child108)
     (served child298)
    )
  )
)
