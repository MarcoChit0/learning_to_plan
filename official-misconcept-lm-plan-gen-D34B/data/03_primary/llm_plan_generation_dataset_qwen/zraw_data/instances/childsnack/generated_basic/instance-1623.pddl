; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 468085

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child383 - child
    bread134 - bread-portion
    content45 - content-portion
    tray418 tray382 tray447 tray62 - tray
    table298 table277 table7 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray418 kitchen)
     (at tray382 kitchen)
     (at tray447 kitchen)
     (at tray62 kitchen)
     (at_kitchen_bread bread134)
     (at_kitchen_content content45)
     (not_allergic_gluten child383)
     (waiting child383 table7)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child383)
    )
  )
)
