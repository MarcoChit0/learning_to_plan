; child-snack task with 2 children and 0.31 gluten factor 
; constant factor of 1.7
; random seed: 760901

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child354 child312 - child
    bread304 bread113 - bread-portion
    content262 content317 - content-portion
    tray180 tray498 tray413 - tray
    table367 table397 table101 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray180 kitchen)
     (at tray498 kitchen)
     (at tray413 kitchen)
     (at_kitchen_bread bread304)
     (at_kitchen_bread bread113)
     (at_kitchen_content content262)
     (at_kitchen_content content317)
     (not_allergic_gluten child312)
     (not_allergic_gluten child354)
     (waiting child354 table367)
     (waiting child312 table397)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child354)
     (served child312)
    )
  )
)
