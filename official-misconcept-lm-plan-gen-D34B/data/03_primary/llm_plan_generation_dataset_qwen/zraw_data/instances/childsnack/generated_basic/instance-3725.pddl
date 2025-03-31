; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 363173

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child124 - child
    bread219 - bread-portion
    content40 - content-portion
    tray306 tray366 tray389 - tray
    table200 table308 table145 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray306 kitchen)
     (at tray366 kitchen)
     (at tray389 kitchen)
     (at_kitchen_bread bread219)
     (at_kitchen_content content40)
     (not_allergic_gluten child124)
     (waiting child124 table308)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child124)
    )
  )
)
