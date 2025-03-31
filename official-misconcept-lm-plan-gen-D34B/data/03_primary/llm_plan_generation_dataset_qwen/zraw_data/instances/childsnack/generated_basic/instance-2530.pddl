; child-snack task with 1 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 351156

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child312 - child
    bread293 - bread-portion
    content324 - content-portion
    tray297 tray497 tray47 - tray
    table428 table47 table70 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray297 kitchen)
     (at tray497 kitchen)
     (at tray47 kitchen)
     (at_kitchen_bread bread293)
     (at_kitchen_content content324)
     (not_allergic_gluten child312)
     (waiting child312 table70)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child312)
    )
  )
)
