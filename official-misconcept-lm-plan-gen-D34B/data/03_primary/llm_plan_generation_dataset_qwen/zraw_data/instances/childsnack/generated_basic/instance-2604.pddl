; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 284927

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child134 - child
    bread180 - bread-portion
    content153 - content-portion
    tray145 tray246 tray468 - tray
    table211 table387 table490 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray145 kitchen)
     (at tray246 kitchen)
     (at tray468 kitchen)
     (at_kitchen_bread bread180)
     (at_kitchen_content content153)
     (not_allergic_gluten child134)
     (waiting child134 table490)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child134)
    )
  )
)
