; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 640039

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child236 - child
    bread319 - bread-portion
    content444 - content-portion
    tray112 tray442 tray386 tray391 - tray
    table305 table333 table56 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray112 kitchen)
     (at tray442 kitchen)
     (at tray386 kitchen)
     (at tray391 kitchen)
     (at_kitchen_bread bread319)
     (at_kitchen_content content444)
     (not_allergic_gluten child236)
     (waiting child236 table56)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child236)
    )
  )
)
