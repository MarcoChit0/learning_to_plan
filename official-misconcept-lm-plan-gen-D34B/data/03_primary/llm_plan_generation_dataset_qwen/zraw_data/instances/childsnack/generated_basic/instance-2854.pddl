; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 34756

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child3 - child
    bread82 - bread-portion
    content8 - content-portion
    tray279 tray486 - tray
    table308 table355 table320 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray279 kitchen)
     (at tray486 kitchen)
     (at_kitchen_bread bread82)
     (at_kitchen_content content8)
     (not_allergic_gluten child3)
     (waiting child3 table308)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child3)
    )
  )
)
