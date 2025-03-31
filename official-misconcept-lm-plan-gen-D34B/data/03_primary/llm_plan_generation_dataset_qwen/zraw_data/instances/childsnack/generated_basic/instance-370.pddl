; child-snack task with 1 children and 0.2 gluten factor 
; constant factor of 1.9
; random seed: 765279

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child166 - child
    bread345 - bread-portion
    content488 - content-portion
    tray490 tray419 tray435 tray366 - tray
    table430 table433 table223 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray490 kitchen)
     (at tray419 kitchen)
     (at tray435 kitchen)
     (at tray366 kitchen)
     (at_kitchen_bread bread345)
     (at_kitchen_content content488)
     (not_allergic_gluten child166)
     (waiting child166 table433)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child166)
    )
  )
)
