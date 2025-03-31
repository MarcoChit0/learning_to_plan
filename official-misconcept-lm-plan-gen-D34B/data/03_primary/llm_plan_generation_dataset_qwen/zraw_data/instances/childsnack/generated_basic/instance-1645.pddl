; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 200661

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child485 - child
    bread400 - bread-portion
    content11 - content-portion
    tray431 tray265 tray65 tray289 - tray
    table81 table126 table455 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray431 kitchen)
     (at tray265 kitchen)
     (at tray65 kitchen)
     (at tray289 kitchen)
     (at_kitchen_bread bread400)
     (at_kitchen_content content11)
     (not_allergic_gluten child485)
     (waiting child485 table455)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child485)
    )
  )
)
